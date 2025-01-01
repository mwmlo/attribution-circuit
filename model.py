import torch
import torch.nn as nn

# n_ctx=42,
#     d_model=56,
#     d_head=28,
#     n_heads=2,
#     d_mlp=56,
#     n_layers=3,
#     attention_dir="bidirectional",  # defaults to "causal"
#     act_fn="relu",
#     d_vocab=len(VOCAB) + 3,  # plus 3 because of end and pad and start token
#     d_vocab_out=2,  # 2 because we're doing binary classification
#     device=device,
#     use_attn_result=True,
#     use_hook_tokens=True,

class BalancedBracketClassifier(nn.Module):
    """
    Balanced bracket classifier based on https://arena-chapter1-transformer-interp.streamlit.app/[1.5.1]_Balanced_Bracket_Classifier
    """

    def __init__(self, d_model=56, n_heads=2, d_mlp=56, n_layers=3):
        super().__init__()

        # Embeddings

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_mlp,
            norm_first=True
        )

        self.encoder_layers = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,

        )

    def forward(self, x):
        pass