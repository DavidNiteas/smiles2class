"""UniMol-based main model (inference and training).

Ported from ``uni_classify.model_tools.main_model`` / ``config`` without the
PyTorch-Lightning training module (which lives in ``training/main``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PretrainedConfig
from unimol_lite.model import UniMolModel
from unimol_lite.model.config import UniMolConfig


class UniClassifyConfig(UniMolConfig):
    model_type = "uniclassify"

    def __init__(
        self,
        encoder_layers=15,
        encoder_embed_dim=512,
        encoder_ffn_embed_dim=2048,
        encoder_attention_heads=64,
        decoder_layers=3,
        decoder_embed_dim=512,
        decoder_ffn_embed_dim=2048,
        decoder_attention_heads=64,
        decoder_token_num=400,
        only_train_decoder=True,
        emb_dropout=0.1,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.0,
        max_seq_len=512,
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        post_ln=False,
        mode="infer",
        delta_pair_repr_norm_loss=-1.0,
        masked_token_loss=-1.0,
        masked_coord_loss=-1.0,
        masked_dist_loss=-1.0,
        **kwargs,
    ):
        self.decoder_layers = decoder_layers
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_token_num = decoder_token_num
        self.only_train_decoder = only_train_decoder
        super().__init__(
            encoder_layers=encoder_layers,
            encoder_embed_dim=encoder_embed_dim,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
            encoder_attention_heads=encoder_attention_heads,
            emb_dropout=emb_dropout,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            pooler_dropout=pooler_dropout,
            max_seq_len=max_seq_len,
            activation_fn=activation_fn,
            pooler_activation_fn=pooler_activation_fn,
            post_ln=post_ln,
            mode=mode,
            delta_pair_repr_norm_loss=delta_pair_repr_norm_loss,
            masked_token_loss=masked_token_loss,
            masked_coord_loss=masked_coord_loss,
            masked_dist_loss=masked_dist_loss,
            **kwargs,
        )

    @classmethod
    def from_json_file(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)


class UniMolModelModify(UniMolModel):
    """Uni-Mol encoder wrapper kept identical to the training-time definition."""

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        return_padding_mask=False,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)
        if src_distance.dim() == 2:
            src_distance = src_distance.unsqueeze(0)
        if src_coord.dim() == 2:
            src_coord = src_coord.unsqueeze(0)
        if src_edge_type.dim() == 2:
            src_edge_type = src_edge_type.unsqueeze(0)

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
            return graph_attn_bias.view(-1, n_node, n_node)

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        return encoder_rep, encoder_pair_rep, padding_mask


class UniClassifyModel(nn.Module):
    """Encoder + per-class query decoder (inference and training)."""

    def __init__(self, configs: UniClassifyConfig):
        super().__init__()
        self.configs = configs
        self.unimol = UniMolModelModify(configs)
        self.class_tokens = nn.Embedding(configs.decoder_token_num, configs.decoder_embed_dim)
        self.class_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=configs.decoder_embed_dim,
                nhead=configs.decoder_attention_heads,
                dim_feedforward=configs.decoder_ffn_embed_dim,
                dropout=configs.dropout,
                batch_first=True,
            ),
            num_layers=configs.decoder_layers,
        )
        self.class_classifier = nn.Linear(configs.decoder_embed_dim, 2)

    @classmethod
    def from_unimol(
        cls,
        unimol_path: str | Path,
        configs: UniClassifyConfig,
        *args,
        **kwargs,
    ) -> "UniClassifyModel":
        """Build the classifier on top of a pre-trained Uni-Mol encoder."""
        unimol = UniMolModelModify.from_pretrained(unimol_path, *args, **kwargs)
        model = cls(configs)
        model.unimol = unimol
        return model

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs,
    ) -> torch.Tensor:
        """Return raw ``(B, decoder_token_num, 2)`` class logits (training)."""
        if self.configs.only_train_decoder:
            with torch.no_grad():
                encoder_rep, _pair_rep, padding_mask = self.unimol(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    encoder_masked_tokens=encoder_masked_tokens,
                    features_only=features_only,
                    classification_head_name=classification_head_name,
                    return_padding_mask=True,
                    **kwargs,
                )
        else:
            encoder_rep, _pair_rep, padding_mask = self.unimol(
                src_tokens,
                src_distance,
                src_coord,
                src_edge_type,
                encoder_masked_tokens=encoder_masked_tokens,
                features_only=features_only,
                classification_head_name=classification_head_name,
                return_padding_mask=True,
                **kwargs,
            )
        tgt = torch.tile(self.class_tokens.weight.unsqueeze(0), (encoder_rep.size(0), 1, 1))
        if padding_mask is not None:
            padding_mask = torch.tile(padding_mask.unsqueeze(1), (1, tgt.size(1), 1))
            padding_mask = torch.repeat_interleave(
                padding_mask, self.configs.decoder_attention_heads, dim=0
            )
        class_embs = self.class_decoder(tgt, encoder_rep, memory_mask=padding_mask)
        return self.class_classifier(class_embs)

    def predict(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(per-class positive probability, CLS embedding)``."""
        self.eval()
        with torch.no_grad():
            encoder_rep, encoder_pair_rep, padding_mask = self.unimol(
                src_tokens,
                src_distance,
                src_coord,
                src_edge_type,
                encoder_masked_tokens=encoder_masked_tokens,
                features_only=features_only,
                classification_head_name=classification_head_name,
                return_padding_mask=True,
                **kwargs,
            )
            tgt = torch.tile(self.class_tokens.weight.unsqueeze(0), (encoder_rep.size(0), 1, 1))
            if padding_mask is not None:
                padding_mask = torch.tile(padding_mask.unsqueeze(1), (1, tgt.size(1), 1))
                padding_mask = torch.repeat_interleave(
                    padding_mask, self.configs.decoder_attention_heads, dim=0
                )
            class_embs = self.class_decoder(tgt, encoder_rep, memory_mask=padding_mask)
            logits = self.class_classifier(class_embs)
            return torch.softmax(logits, dim=-1)[:, :, 1], encoder_rep[:, 0, :]


def load_main_model(model_dir: str | Path, device: str = "cpu") -> UniClassifyModel:
    model_dir = Path(model_dir)
    cfg = UniClassifyConfig.from_json_file(str(model_dir / "config.json"))
    model = UniClassifyModel(cfg)
    state = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected tensors in {model_dir}: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"missing tensors in {model_dir}: {missing[:5]}")
    return model.to(device).eval()
