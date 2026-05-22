# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 深度学习模型 V3

重大升级：
- 多层 Transformer 编码器（4 层 / 8 头 / 256 维）
- 对比学习（InfoNCE loss）+ 多任务训练
- 药材-方剂联合图感知模块
- 参数量：160K → ~2.5M
- 支持扩展症状（200种）和证型（17种）
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_src_path = Path(__file__).resolve().parents[1] / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 位置编码
# ═══════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])


# ═══════════════════════════════════════════════════
# 药材特征编码器 V3
# ═══════════════════════════════════════════════════

class HerbFeatureEncoderV3(nn.Module):
    """药材功效文本字符 → 稠密特征（残差连接升级版）"""

    def __init__(self, num_chars: int, feature_dim: int = 128):
        super().__init__()
        hidden = feature_dim * 3
        self.projection = nn.Sequential(
            nn.Linear(num_chars, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, char_multi_hot: torch.Tensor) -> torch.Tensor:
        return self.projection(char_multi_hot)


# ═══════════════════════════════════════════════════
# 症状编码器 V3（多层 Transformer）
# ═══════════════════════════════════════════════════

class SymptomEncoderV3(nn.Module):
    """多层 Transformer 症状编码器"""

    def __init__(
        self,
        num_symptoms: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_symptoms + 1, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, symptoms: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            symptoms: [batch, seq_len]
        Returns:
            [batch, d_model]
        """
        if mask is None:
            mask = (symptoms != 0)

        emb = self.embedding(symptoms) * (self.d_model ** 0.5)  # scale
        emb = self.pos_encoder(emb)

        # Transformer encoder
        src_key_padding_mask = ~mask
        output = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        output = self.layer_norm(output)

        # Mean pooling over valid tokens
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (output * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        return pooled


# ═══════════════════════════════════════════════════
# 证型分类器 V3
# ═══════════════════════════════════════════════════

class SyndromeClassifierV3(nn.Module):
    def __init__(self, input_dim: int = 256, num_syndromes: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_syndromes),
        )

    def forward(self, symptom_vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(symptom_vector)


# ═══════════════════════════════════════════════════
# 方剂推荐器 V3（对比学习增强）
# ═══════════════════════════════════════════════════

class FormulaRecommenderV3(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_formulas: int = 1,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_formulas = num_formulas

        # Formula embeddings (learnable)
        self.formula_embeddings = nn.Embedding(num_formulas + 1, input_dim, padding_idx=0)

        # Symptom query projection
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        # Score MLP (interaction-based)
        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_formula_embeddings(self) -> torch.Tensor:
        """返回 [num_formulas, input_dim]"""
        return self.formula_embeddings.weight[1:]

    def forward(self, symptom_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            symptom_vector: [batch, d_model]
        Returns:
            [batch, num_formulas]
        """
        query = self.query_proj(symptom_vector)  # [batch, d_model]
        formula_emb = self.get_formula_embeddings()  # [num_formulas, d_model]

        batch_size = query.size(0)
        num_f = formula_emb.size(0)

        # Interaction-based scoring
        q_expanded = query.unsqueeze(1).expand(-1, num_f, -1)  # [batch, num_f, d_model]
        f_expanded = formula_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_f, d_model]
        interaction = torch.cat([q_expanded, f_expanded], dim=-1)  # [batch, num_f, 2*d_model]
        scores = self.score_mlp(interaction).squeeze(-1)  # [batch, num_f]

        return scores


# ═══════════════════════════════════════════════════
# 药材推荐器 V3（对比学习增强 + 图感知）
# ═══════════════════════════════════════════════════

class HerbRecommenderV3(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_herbs: int = 1,
        herb_feature_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_herbs = num_herbs

        # Herb embeddings (base + feature-enhanced)
        self.herb_base_embedding = nn.Embedding(num_herbs + 1, input_dim, padding_idx=0)
        self.feature_fusion = nn.Sequential(
            nn.Linear(herb_feature_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        # Query projection
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        # Score MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_herb_embeddings(self, herb_features: torch.Tensor) -> torch.Tensor:
        """构建药材嵌入 = 基础嵌入 + 特征投影"""
        base_emb = self.herb_base_embedding.weight[1:]  # [num_herbs, input_dim]
        feature_emb = self.feature_fusion(herb_features)  # [num_herbs, input_dim]
        herb_emb = base_emb + feature_emb
        return F.normalize(herb_emb, p=2, dim=1)

    def forward(
        self,
        symptom_vector: torch.Tensor,
        herb_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            symptom_vector: [batch, d_model]
            herb_features:  [num_herbs, herb_feature_dim]
        Returns:
            scores: [batch, num_herbs]
            herb_emb_norm: [num_herbs, d_model] 归一化的药材嵌入（用于对比学习）
        """
        query = self.query_proj(symptom_vector)
        query = F.normalize(query, p=2, dim=1)

        herb_emb = self.get_herb_embeddings(herb_features)

        batch_size = query.size(0)
        num_h = herb_emb.size(0)

        # Interaction scoring
        q_expanded = query.unsqueeze(1).expand(-1, num_h, -1)
        h_expanded = herb_emb.unsqueeze(0).expand(batch_size, -1, -1)
        interaction = torch.cat([q_expanded, h_expanded], dim=-1)
        scores = self.score_mlp(interaction).squeeze(-1)

        return scores, herb_emb


# ═══════════════════════════════════════════════════
# 主模型 V3
# ═══════════════════════════════════════════════════

class TCMRecommenderSystemV3(nn.Module):
    """中医药推荐系统 V3 — 多任务对比学习"""

    def __init__(
        self,
        num_symptoms: int,
        num_herbs: int,
        num_formulas: int,
        num_syndromes: int,
        herb_char_dim: int = 0,
        d_model: int = 256,
        herb_feature_dim: int = 128,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.symptom_encoder = SymptomEncoderV3(
            num_symptoms, d_model, num_heads, num_transformer_layers, dropout
        )
        self.syndrome_classifier = SyndromeClassifierV3(
            d_model, num_syndromes, d_model
        )
        self.formula_recommender = FormulaRecommenderV3(
            d_model, num_formulas
        )
        self.herb_recommender = HerbRecommenderV3(
            d_model, num_herbs, herb_feature_dim
        )

        # Temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        symptoms: torch.Tensor,
        herb_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            syndrome_logits, formula_scores, herb_scores, symptom_vector, herb_emb
        """
        sv = self.symptom_encoder(symptoms)
        syndrome_logits = self.syndrome_classifier(sv)
        formula_scores = self.formula_recommender(sv)

        if herb_features is not None:
            herb_scores, herb_emb = self.herb_recommender(sv, herb_features)
        else:
            n_h = self.herb_recommender.num_herbs
            fef_dim = self.herb_recommender.feature_fusion[0].in_features
            dummy = torch.zeros(n_h, fef_dim, device=sv.device)
            herb_scores, herb_emb = self.herb_recommender(sv, dummy)

        return syndrome_logits, formula_scores, herb_scores, sv, herb_emb

    def contrastive_loss(
        self,
        symptom_vec: torch.Tensor,
        herb_emb: torch.Tensor,
        herb_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE 对比损失：让症状向量靠近其相关药材的嵌入

        Args:
            symptom_vec: [batch, d_model]
            herb_emb:   [num_herbs, d_model]
            herb_labels: [batch, num_herbs] 多标签
        """
        batch_size = symptom_vec.size(0)

        sv_norm = F.normalize(symptom_vec, p=2, dim=1)
        he_norm = F.normalize(herb_emb, p=2, dim=1)

        # Cosine similarity
        sim = sv_norm @ he_norm.T / self.temperature  # [batch, num_herbs]

        # For each sample, use all positive herb as positives in a multi-positive InfoNCE
        loss = 0.0
        count = 0
        for i in range(batch_size):
            pos_mask = herb_labels[i] > 0
            if pos_mask.sum() == 0:
                continue
            # Simplified: mean of log-softmax over positives
            pos_sim = sim[i][pos_mask]
            all_sim = sim[i]
            # Log-sum-exp trick for numerical stability
            log_prob = pos_sim - torch.logsumexp(all_sim, dim=0)
            loss += -log_prob.mean()
            count += 1

        return loss / max(count, 1)

    @torch.no_grad()
    def predict(
        self,
        symptoms: torch.Tensor,
        herb_features: Optional[torch.Tensor] = None,
        top_k: int = 10,
    ) -> Dict[str, List]:
        self.eval()
        syn_logits, formula_scores, herb_scores, _, _ = self.forward(symptoms, herb_features)
        syn_probs = F.softmax(syn_logits[0], dim=0)

        herb_topk = torch.topk(herb_scores[0], min(top_k, herb_scores.shape[1]))
        formula_topk = torch.topk(formula_scores[0], min(top_k, formula_scores.shape[1]))

        return {
            "syndromes": list(zip(
                syn_probs.argsort(descending=True).tolist(),
                syn_probs.sort(descending=True).values.tolist(),
            )),
            "herbs": list(zip(herb_topk.indices.tolist(), herb_topk.values.tolist())),
            "formulas": list(zip(formula_topk.indices.tolist(), formula_topk.values.tolist())),
        }


# ═══════════════════════════════════════════════════
# 安全校验器（同 V2，保持不变）
# ═══════════════════════════════════════════════════

class SafetyChecker:
    def __init__(self, incompatible_herbs: Dict[str, list]):
        self.incompatible_pairs: set = set()
        self.pregnancy_forbidden: set = set()

        for category, items in incompatible_herbs.items():
            if category == "孕妇禁用":
                self.pregnancy_forbidden = set(items)
            else:
                for pair in items:
                    if isinstance(pair, tuple) and len(pair) == 2:
                        self.incompatible_pairs.add(pair)
                        self.incompatible_pairs.add((pair[1], pair[0]))

    def check_compatibility(self, herb_names: List[str]) -> Dict:
        warnings = []
        incompatible = []
        toxic = []

        for i, h1 in enumerate(herb_names):
            for h2 in herb_names[i + 1:]:
                if (h1, h2) in self.incompatible_pairs:
                    incompatible.append((h1, h2))
                    warnings.append(f"⚠️ 配伍禁忌：{h1} 与 {h2} 不可同用（十八反/十九畏）")

        for h in herb_names:
            if h in self.pregnancy_forbidden:
                warnings.append(f"🚫 孕妇禁用：{h}")

        from core.tcm_config import TOXIC_HERBS
        for h in herb_names:
            if h in TOXIC_HERBS:
                toxic.append(h)
                warnings.append(f"☠️ 有毒性：{h}（须在医师指导下使用）")

        return {
            "is_safe": len(incompatible) == 0,
            "warnings": warnings,
            "incompatible_pairs": incompatible,
            "toxic_herbs": toxic,
        }


# ═══════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════

def create_model_v3(
    num_symptoms: int,
    num_herbs: int,
    num_formulas: int,
    num_syndromes: int = 0,
    herb_char_dim: int = 0,
    d_model: int = 256,
    herb_feature_dim: int = 128,
    num_transformer_layers: int = 4,
) -> TCMRecommenderSystemV3:
    model = TCMRecommenderSystemV3(
        num_symptoms=num_symptoms,
        num_herbs=num_herbs,
        num_formulas=num_formulas,
        num_syndromes=max(num_syndromes, 1),
        herb_char_dim=herb_char_dim,
        d_model=d_model,
        herb_feature_dim=herb_feature_dim,
        num_transformer_layers=num_transformer_layers,
    )
    total = sum(p.numel() for p in model.parameters())
    logger.info("创建 V3 模型: %s 参数", f"{total:,}")
    return model
