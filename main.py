import json
from itertools import takewhile, dropwhile
import einops
from torch import tensor
import torch as t
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from brackets_datasets import BracketsDataset, SimpleTokenizer
from model import compute_integrated_gradients_layer

from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import HookPoint
from captum.attr import LayerIntegratedGradients

import plotly_utils
from plotly_utils import bar, hist

device = t.device("mps")

# Balanced bracket classifier

VOCAB = "()"

cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional",  # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB) + 3,  # plus 3 because of end and pad and start token
    d_vocab_out=2,  # 2 because we're doing binary classification
    device=device,
    use_attn_result=True,
    use_hook_tokens=True,
)

model = HookedTransformer(cfg).eval()
print(model)
tokenizer = SimpleTokenizer("()")

state_dict = t.load("brackets_model_state_dict.pt", map_location=device)
model.load_state_dict(state_dict)

def add_perma_hooks_to_mask_pad_tokens(model: HookedTransformer, pad_token: int) -> HookedTransformer:
    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(tokens, hook) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores,
        hook,
    ) -> None:
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model

model.reset_hooks(including_permanent=True)
model = add_perma_hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)

print("Loaded model")

N_SAMPLES = 5000
with open("brackets_data.json") as f:
    data_tuples = json.load(f)
    print(f"loaded {len(data_tuples)} examples, using {N_SAMPLES}")
    data_tuples = data_tuples[:N_SAMPLES]

data = BracketsDataset(data_tuples).to(device)
data_mini = BracketsDataset(data_tuples[:100]).to(device)

# def run_model_on_data(
#     model: HookedTransformer, data: BracketsDataset, batch_size: int = 200
# ):
#     """Return probability that each example is balanced"""
#     all_logits = []
#     for i in range(0, len(data.strs), batch_size):
#         toks = data.toks[i : i + batch_size]
#         logits = model(toks)[:, 0]
#         all_logits.append(logits)
#     all_logits = t.cat(all_logits)
#     assert all_logits.shape == (len(data), 2)
#     return all_logits

# test_set = data
# n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
# print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")

# applying integrated gradients on model

def predict(x):
    logits = model(x)[:, 0]
    return logits.softmax(-1)[:, 0]

input = tokenizer.tokenize("()()")
mask = np.isin(input, [tokenizer.START_TOKEN, tokenizer.END_TOKEN])
baseline = input * mask + tokenizer.PAD_TOKEN * (1 - mask)

print(f"Input shape (tokens) {input.shape}")

target_layers = [
    model.embed, 
    model.blocks[0].ln1, 
    model.blocks[0].attn, 
    model.blocks[0].ln2,
    model.blocks[0].mlp, 
    model.blocks[1].ln1, 
    model.blocks[1].attn, 
    model.blocks[1].ln2,
    model.blocks[1].mlp, 
    model.blocks[2].ln1, 
    model.blocks[2].attn, 
    model.blocks[2].ln2,
    model.blocks[2].mlp, 
    model.ln_final
]
layer_names = [
    "embed",
    "0-ln1",
    "0-attn",
    "0-ln2",
    "0-mlp",
    "1-ln1",
    "1-attn",
    "1-ln2",
    "1-mlp",
    "2-ln1",
    "2-attn",
    "2-ln2",
    "2-mlp",
    "lnfinal"
]


def compute_layer_to_output_attributions(target_layer):
    ig_embed = LayerIntegratedGradients(predict, target_layer)
    attributions, approximation_error = ig_embed.attribute(inputs=input,
                                                    baselines=baseline,
                                                    return_convergence_delta=True)

    # print(f"Attributions (shape {embed_attributions.shape}): \n{embed_attributions}")
    # print("\nError:", approximation_error.item())
    return attributions

all_layers = list(model.hook_dict.keys())

def get_pre_layers(to_layer_name):
    pre_layers = list(takewhile(lambda x: x != to_layer_name, all_layers))
    # Include to_layer in list
    pre_layers.append(all_layers[len(pre_layers)])
    return pre_layers


def compute_layer_to_layer_attributions(from_layer, stop_layer_index):
    
    def run_fn(x):
        return model.forward(x, stop_at_layer=stop_layer_index)
    
    ig_embed = LayerIntegratedGradients(run_fn, from_layer)
    attributions, approximation_error = ig_embed.attribute(inputs=input,
                                                    baselines=baseline,
                                                    target=(0,0),
                                                    return_convergence_delta=True)

    # print(f"Attributions (shape {embed_attributions.shape}): \n{embed_attributions}")
    # print("\nError:", approximation_error.item())
    return attributions


def display_attributions(attributions, from_layer_name, to_layer_name, averaged=False):
    attrs = attributions[0].numpy() # one input token
    if averaged:
        attrs = np.mean(attrs, axis=1).reshape((-1, 1))
    ax = sns.heatmap(attrs, cmap="coolwarm", linewidths=0.5, center=0)
    plt.title(f"{from_layer_name}->{to_layer_name}")
    plt.savefig(f"{from_layer_name}_{to_layer_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# attrs_0_attn_in = compute_layer_to_layer_attributions(model.blocks[0].ln1, 1)
# attrs_0_mlp_in = compute_layer_to_layer_attributions(model.blocks[0].ln2, 1)
# display_attributions(attrs_0_attn_in, "0-attn-in", "0-output")
# display_attributions(attrs_0_mlp_in, "0-mlp-in", "0-output")

# display_attributions(attrs_0_attn_in - attrs_0_mlp_in, "0-attn-in ISOLATED", "0-output")


# for layer, name in zip(target_layers, layer_names):
#     attrs = compute_layer_to_output_attributions(layer)
#     display_attributions(attrs, name, "output(avg)", averaged=True)

# same as   compute_layer_to_output_attributions(model.embed, "embed")
# ==        compute_layer_to_layer_attributions(model.embed, None, "embed", "output")

# compute_layer_to_layer_attributions(model.blocks[0].ln1, 1, "0-attn-in", "0-output")
# compute_layer_to_layer_attributions(model.blocks[0].ln2, 1, "0-mlp-in", "0-output")
# compute_layer_to_layer_attributions(model.blocks[0].ln1, 3, "0-attn-in", "output")
# compute_layer_to_layer_attributions(model.blocks[2].ln1, 3, "2-attn-in", "2-output")
# compute_layer_to_layer_attributions(model.blocks[2].attn, 3, "2-attn-out", "2-output")

from arena_solutions import get_out_by_neuron_in_20_dir_less_memory, is_balanced_vectorized_return_both

total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)

failure_types_dict = {
    "both failures": negative_failure & total_elevation_failure,
    "just neg failure": negative_failure & ~total_elevation_failure,
    "just total elevation failure": ~negative_failure & total_elevation_failure,
    "balanced": ~negative_failure & ~total_elevation_failure,
}

for layer in range(2):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_unbalanced_dir = get_out_by_neuron_in_20_dir_less_memory(model, data, layer)[utils.to_numpy(data.starts_open), :]
    # Plot neurons' activations
    plotly_utils.plot_neurons(neurons_in_unbalanced_dir, model, data, failure_types_dict, layer, renderer="browser")
