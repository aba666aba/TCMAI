# Copyright (C) 2026 中医药智能问答系统 (TCMAI)
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。

"""TCMNet 深度学习模型 — Transformer + 双塔CNN + 交叉注意力 + 多任务学习"""

from typing import Optional

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

logger = logging.getLogger(__name__)


# 基础组件

class PreLayerNormTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder Layer

    与标准 Post-LN 不同，这里把 LayerNorm 放在 attention/FFN 之前（Pre-LN），
    残差连接后不再归一化。好处是梯度流更稳定，深层网络不容易梯度消失，
    训练初期收敛更快。这也是 GPT-2、BERT-wwm 等后续模型采用的方案。
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu"):
        """
        Args:
            d_model: 每个 token 的特征维度
            nhead: 多头注意力的头数（d_model 必须能被 nhead 整除）
            dim_feedforward: FFN 中间层维度，通常是 d_model 的 4 倍
            dropout: 注意力和 FFN 的 dropout 概率
            activation: FFN 激活函数，"gelu" 或 "relu"
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout_act = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: 输入序列 [batch, seq_len, d_model]
            src_mask: 注意力掩码（本项目未使用，预留接口）
            src_key_padding_mask: padding 位置标记 [batch, seq_len]，True 表示该位置是 padding

        Returns:
            与输入同维度的输出 [batch, seq_len, d_model]
        """
        # Pre-LN: 先归一化再做 attention，残差连接在归一化之前
        residual = src
        src_normed = self.norm1(src)
        attn_out, _ = self.self_attn(src_normed, src_normed, src_normed,
                                      key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout1(attn_out)

        # Pre-LN + FFN: 两层线性变换 + 激活，典型瓶颈结构
        residual = src
        src_normed = self.norm2(src)
        ffn_out = self.linear2(self.dropout_act(self.activation(self.linear1(src_normed))))
        src = residual + self.dropout2(ffn_out)
        return src


class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:, :x.size(1)]


class AttentivePooling(nn.Module):
    """注意力池化：加权平均序列 → 固定维度"""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            mask: [B, L] True=valid
        Returns:
            [B, D]
        """
        scores = self.attn(x).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        return (x * weights).sum(dim=1)


# 症状编码器

class SymptomEncoder(nn.Module):
    """多层Transformer症状编码器"""

    def __init__(self, num_symptoms: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_symptoms + 1, d_model, padding_idx=0)
        self.pos_encoder = LearnedPositionalEncoding(d_model)
        self.dropout_emb = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            PreLayerNormTransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pooling = AttentivePooling(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.constant_(self.embedding.weight[0], 0)

    def forward(self, symptom_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            symptom_ids: [B, L], 1-based, 0=padding
            mask: [B, L], True=valid
        Returns:
            pooled: [B, D]
            sequence: [B, L, D]
        """
        if mask is None:
            mask = symptom_ids != 0

        x = self.embedding(symptom_ids) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        x = self.dropout_emb(x)

        key_padding_mask = ~mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        pooled = self.pooling(x, mask)
        return pooled, x


# 药材编码器 (双塔)

class HerbEncoder(nn.Module):
    """双塔药材编码器：文本塔 + 属性塔"""

    def __init__(self, num_chars: int, num_categories: int = 30,
                 d_model: int = 512, text_dim: int = 256, meta_dim: int = 64):
        super().__init__()
        # 文本塔 (CharCNN → 文本特征)
        self.text_encoder = nn.Sequential(
            nn.Linear(num_chars, text_dim * 2),
            nn.LayerNorm(text_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(text_dim * 2, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 属性塔 (category + nature + flavor + meridian)
        self.category_emb = nn.Embedding(num_categories + 1, meta_dim, padding_idx=0)
        self.nature_emb = nn.Embedding(8, meta_dim)    # 0=padding, 1-5=natures, 6-7=unknown
        self.flavor_emb = nn.Embedding(8, meta_dim)     # 0=padding, 1-5=flavors, 6-7=unknown
        meta_total = meta_dim * 3
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_total, d_model),
            nn.LayerNorm(d_model),
        )

        # 融合门
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, char_multihot: torch.Tensor, category_idx: torch.Tensor,
                nature_idx: torch.Tensor, flavor_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_multihot: [num_herbs, num_chars]
            category_idx: [num_herbs] category id
            nature_idx: [num_herbs] nature id
            flavor_idx: [num_herbs] flavor id
        Returns:
            [num_herbs, d_model]
        """
        text_feat = self.text_encoder(char_multihot)  # [H, D]
        meta = torch.cat([
            self.category_emb(category_idx),
            self.nature_emb(nature_idx),
            self.flavor_emb(flavor_idx),
        ], dim=-1)
        meta_feat = self.meta_proj(meta)  # [H, D]

        gate = self.gate(torch.cat([text_feat, meta_feat], dim=-1))
        return gate * text_feat + (1 - gate) * meta_feat


# 交叉注意力模块

class CrossAttentionRanker(nn.Module):
    """症状-药材交叉注意力排序器"""

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, symptom_repr: torch.Tensor,
                item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            symptom_repr: [B, D]  症状表征
            item_embeddings: [num_items, D]  药材/方剂嵌入
        Returns:
            [B, num_items] 分数
        """
        B = symptom_repr.size(0)
        N = item_embeddings.size(0)

        # Query=B个症状, Key/Value=N个候选项
        q = symptom_repr.unsqueeze(1)  # [B, 1, D]
        k = item_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        v = k

        attn_out, _ = self.cross_attn(q, k, v)  # [B, 1, D]
        attn_out = self.norm(attn_out + q)

        # 逐项评分
        combined = attn_out.expand(-1, N, -1) * item_embeddings.unsqueeze(0)  # [B, N, D]
        scores = self.score_proj(combined).squeeze(-1)  # [B, N]
        return scores


# 多任务头

class SyndromeHead(nn.Module):
    def __init__(self, d_model: int = 512, num_syndromes: int = 18, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_syndromes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 主模型

class TCMNet(nn.Module):
    """中医药推荐系统模型"""

    def __init__(self, num_symptoms: int, num_herbs: int, num_formulas: int,
                 num_syndromes: int, num_chars: int, num_categories: int = 30,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # 编码器
        self.symptom_encoder = SymptomEncoder(
            num_symptoms, d_model, nhead, num_layers, d_model * 4, dropout)
        self.herb_encoder = HerbEncoder(
            num_chars, num_categories, d_model, text_dim=256, meta_dim=64)

        # 方剂嵌入 (可学习)
        self.formula_embedding = nn.Embedding(num_formulas + 1, d_model, padding_idx=0)

        # 排序器
        self.herb_ranker = CrossAttentionRanker(d_model, nhead, dropout)
        self.formula_ranker = CrossAttentionRanker(d_model, nhead, dropout)

        # 分类头
        self.syndrome_head = SyndromeHead(d_model, num_syndromes, dropout)

        # 投影层 (用于对比学习)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 256),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode_symptoms(self, symptom_ids: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled, _ = self.symptom_encoder(symptom_ids, mask)
        return pooled

    def encode_herbs(self, char_mh: torch.Tensor, cat_idx: torch.Tensor,
                     nat_idx: torch.Tensor, flv_idx: torch.Tensor) -> torch.Tensor:
        return self.herb_encoder(char_mh, cat_idx, nat_idx, flv_idx)

    def forward(self, symptom_ids: torch.Tensor, symptom_mask: torch.Tensor,
                herb_char: torch.Tensor, herb_cat: torch.Tensor,
                herb_nat: torch.Tensor, herb_flv: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            syndrome_logits: [B, num_syndromes]
            herb_scores: [B, num_herbs]
            formula_scores: [B, num_formulas]
            symptom_vec: [B, D]  (for contrastive loss)
            herb_vec: [H, D]      (for contrastive loss)
        """
        sv, _ = self.symptom_encoder(symptom_ids, symptom_mask > 0.5)
        hv = self.herb_encoder(herb_char, herb_cat, herb_nat, herb_flv)

        syndrome_logits = self.syndrome_head(sv)

        herb_scores = self.herb_ranker(sv, hv)
        formula_emb = self.formula_embedding.weight[1:]  # [F, D]
        formula_scores = self.formula_ranker(sv, formula_emb)

        return syndrome_logits, herb_scores, formula_scores, sv, hv

    def contrastive_loss(self, symptom_vec: torch.Tensor, herb_vec: torch.Tensor,
                         herb_labels: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Supervised Contrastive Loss (多正类版本)
        """
        tau = temperature if temperature is not None else self.temperature.abs()

        sv = F.normalize(self.projection(symptom_vec), dim=1)
        hv = F.normalize(self.projection(herb_vec), dim=1)

        sim = sv @ hv.T / tau.clamp(min=0.01)  # [B, H]

        loss = 0.0
        count = 0
        for i in range(sim.size(0)):
            pos_mask = herb_labels[i] > 0.5
            if pos_mask.sum() == 0:
                continue
            pos_sim = sim[i][pos_mask]
            log_prob = pos_sim - torch.logsumexp(sim[i], dim=0)
            loss -= log_prob.mean()
            count += 1

        return loss / max(count, 1)

    @torch.no_grad()
    def predict(self, symptom_ids: torch.Tensor, symptom_mask: torch.Tensor,
                herb_char: torch.Tensor, herb_cat: torch.Tensor,
                herb_nat: torch.Tensor, herb_flv: torch.Tensor,
                top_k: int = 10) -> dict[str, list]:
        self.eval()
        syn_logits, herb_scores, formula_scores, _, _ = self.forward(
            symptom_ids, symptom_mask, herb_char, herb_cat, herb_nat, herb_flv)

        syn_probs = F.softmax(syn_logits[0], dim=-1)
        h_topk = torch.topk(herb_scores[0], min(top_k, herb_scores.shape[1]))
        f_topk = torch.topk(formula_scores[0], min(top_k, formula_scores.shape[1]))

        return {
            "syndromes": list(zip(syn_probs.argsort(descending=True).tolist(),
                                  syn_probs.sort(descending=True).values.tolist(), strict=False)),
            "herbs": list(zip(h_topk.indices.tolist(), h_topk.values.tolist(), strict=False)),
            "formulas": list(zip(f_topk.indices.tolist(), f_topk.values.tolist(), strict=False)),
        }


# 工厂函数

def create_model(**kwargs) -> TCMNet:
    model = TCMNet(**kwargs)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型: %s 总参数 (%s 可训练)", f"{total:,}", f"{trainable:,}")
    return model
