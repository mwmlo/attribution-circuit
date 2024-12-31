import json
import tqdm
import einops
from torch import Tensor, nn
import torch as t
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl

from brackets_datasets import BracketsDataset, SimpleTokenizer

from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig, utils
from captum.attr import IntegratedGradients

device = t.device("mps")
t.set_grad_enabled(False)

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

N_SAMPLES = 5000
with open("brackets_data.json") as f:
    data_tuples = json.load(f)
    print(f"loaded {len(data_tuples)} examples, using {N_SAMPLES}")
    data_tuples = data_tuples[:N_SAMPLES]

data = BracketsDataset(data_tuples).to(device)
data_mini = BracketsDataset(data_tuples[:100]).to(device)

def run_model_on_data(
    model: HookedTransformer, data: BracketsDataset, batch_size: int = 200
):
    """Return probability that each example is balanced"""
    all_logits = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        logits = model(toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits

test_set = data
n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")