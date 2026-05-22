# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 训练脚本 V3

升级：
- 使用 V3 模型（多层 Transformer + 对比学习）
- 扩展数据集（200 症状 + 17 证型 + 证型关联增强）
- 多任务损失：分类 + 排序 + 对比
- 更大的批次 / 更多 epoch
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_loader import TCMDataLoader
from tcm_recommender.model_v3 import (
    TCMRecommenderSystemV3,
    HerbFeatureEncoderV3,
    SafetyChecker,
    create_model_v3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 增强数据集 V3
# ═══════════════════════════════════════════════════

class TCMDatasetV3(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        max_symptoms: int = 30,
    ):
        self.samples = samples
        self.max_symptoms = max_symptoms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        syms = sample["symptoms"][:self.max_symptoms]
        syms = syms + [0] * (self.max_symptoms - len(syms))

        return {
            "symptoms": torch.tensor(syms, dtype=torch.long),
            "herb_labels": torch.tensor(sample["herb_labels"], dtype=torch.float),
            "formula_label": torch.tensor(sample["formula_label"], dtype=torch.long),
            "syndrome_label": torch.tensor(sample["syndrome_label"], dtype=torch.long),
        }


def build_enhanced_training_data(
    data_loader: TCMDataLoader,
    extended_symptoms: Optional[List[str]] = None,
    extended_syndromes: Optional[Dict] = None,
) -> Tuple[List[Dict], int, int]:
    """
    构建增强训练数据

    三个来源：
    1. 方剂样本：从方剂组成提取药材，从主治提取症状
    2. 药材样本：从药材功效+主治匹配症状
    3. 证型增强样本：从证型定义生成症状→药材/方剂关联
    """
    from core.tcm_config import SYMPTOM_LIST, SYNDROME_RULES

    all_symptoms = list(extended_symptoms) if extended_symptoms else list(SYMPTOM_LIST)
    symptom_to_idx = {s: i + 1 for i, s in enumerate(all_symptoms)}  # 1-based

    all_syndromes = dict(extended_syndromes) if extended_syndromes else dict(SYNDROME_RULES)
    syndrome_names = list(all_syndromes.keys())
    num_syndromes = len(syndrome_names)

    samples: List[Dict] = []

    # ── 来源1：方剂样本 ──
    n_formula = 0
    for formula_id, formula in data_loader.formulas.items():
        herb_indices = data_loader._parse_composition(formula.get("composition", ""))
        if not herb_indices or len(herb_indices) < 1:
            continue

        eff_text = " ".join([
            formula.get("functions", "") or "",
            formula.get("indications", "") or "",
        ])
        # 也收集药材功效
        for idx in herb_indices:
            herb_id = data_loader.idx_to_herb_id.get(idx)
            if herb_id and herb_id in data_loader.herbs:
                h = data_loader.herbs[herb_id]
                for f in ["efficacy", "indications"]:
                    if h.get(f):
                        eff_text += " " + h[f]

        symptom_ids = _match_symptoms_in_text_v2(eff_text, all_symptoms)
        if not symptom_ids:
            continue

        syndrome_id = _infer_syndrome_v2(symptom_ids, all_symptoms, all_syndromes)
        formula_idx = data_loader.formula_id_to_idx.get(formula_id, 0)

        num_herbs = data_loader.num_herbs
        herb_labels = np.zeros(num_herbs, dtype=np.float32)
        for hi in herb_indices:
            if 0 < hi <= num_herbs:
                herb_labels[hi - 1] = 1.0

        samples.append({
            "symptoms": symptom_ids,
            "herb_labels": herb_labels.tolist(),
            "formula_label": formula_idx,
            "syndrome_label": syndrome_id,
        })
        n_formula += 1

    logger.info("方剂样本: %s", n_formula)

    # ── 来源2：药材样本 ──
    n_herb = 0
    for herb_id, herb in data_loader.herbs.items():
        herb_idx = data_loader.herb_id_to_idx.get(herb_id)
        if herb_idx is None:
            continue

        eff_text = " ".join([
            herb.get("efficacy", "") or "",
            herb.get("indications", "") or "",
        ])
        symptom_ids = _match_symptoms_in_text_v2(eff_text, all_symptoms)
        if not symptom_ids:
            continue

        syndrome_id = _infer_syndrome_v2(symptom_ids, all_symptoms, all_syndromes)
        num_herbs = data_loader.num_herbs
        herb_labels = np.zeros(num_herbs, dtype=np.float32)
        herb_labels[herb_idx - 1] = 1.0

        samples.append({
            "symptoms": symptom_ids,
            "herb_labels": herb_labels.tolist(),
            "formula_label": 0,
            "syndrome_label": syndrome_id,
        })
        n_herb += 1

    logger.info("药材样本: %s", n_herb)

    # ── 来源3：证型增强样本（从证型定义直接生成） ──
    n_syndrome_aug = 0
    if extended_syndromes:
        for syn_name, syn_info in extended_syndromes.items():
            syn_symptoms = syn_info.get("symptoms", [])
            syn_herbs = syn_info.get("recommend_herbs", [])
            syn_formulas = syn_info.get("recommend_formulas", [])

            symptom_ids = []
            for s in syn_symptoms:
                if s in symptom_to_idx:
                    symptom_ids.append(symptom_to_idx[s])

            if not symptom_ids:
                continue

            # 找到对应的 herb indices
            herb_indices = []
            for h_name in syn_herbs:
                h_id = data_loader.herb_name_to_db_id.get(h_name)
                if h_id and h_id in data_loader.herb_id_to_idx:
                    herb_indices.append(data_loader.herb_id_to_idx[h_id])

            # 混合：部分症状子集 + 全部药材 → 多个样本
            if len(symptom_ids) >= 3:
                # 生成子集变体增加样本多样性
                for _ in range(min(3, len(symptom_ids))):
                    subset_size = max(2, len(symptom_ids) // 2)
                    subset = list(np.random.RandomState(42 + n_syndrome_aug).choice(
                        symptom_ids, size=min(subset_size, len(symptom_ids)), replace=False
                    ))
                    syn_idx = syndrome_names.index(syn_name) + 1 if syn_name in syndrome_names else 0
                    num_herbs = data_loader.num_herbs
                    hl = np.zeros(num_herbs, dtype=np.float32)
                    for hi in herb_indices:
                        if 0 < hi <= num_herbs:
                            hl[hi - 1] = 1.0

                    # 找到方剂索引
                    formula_label = 0
                    if syn_formulas:
                        f_name = syn_formulas[0]
                        for fid, finfo in data_loader.formulas.items():
                            if finfo.get("name") == f_name and fid in data_loader.formula_id_to_idx:
                                formula_label = data_loader.formula_id_to_idx[fid]
                                break

                    samples.append({
                        "symptoms": subset,
                        "herb_labels": hl.tolist(),
                        "formula_label": formula_label,
                        "syndrome_label": syn_idx,
                    })
                    n_syndrome_aug += 1

            # 完整症状版本
            if len(symptom_ids) >= 2:
                syn_idx = syndrome_names.index(syn_name) + 1 if syn_name in syndrome_names else 0
                num_herbs = data_loader.num_herbs
                hl = np.zeros(num_herbs, dtype=np.float32)
                for hi in herb_indices:
                    if 0 < hi <= num_herbs:
                        hl[hi - 1] = 1.0

                formula_label = 0
                if syn_formulas:
                    f_name = syn_formulas[0]
                    for fid, finfo in data_loader.formulas.items():
                        if finfo.get("name") == f_name and fid in data_loader.formula_id_to_idx:
                            formula_label = data_loader.formula_id_to_idx[fid]
                            break

                samples.append({
                    "symptoms": symptom_ids,
                    "herb_labels": hl.tolist(),
                    "formula_label": formula_label,
                    "syndrome_label": syn_idx,
                })
                n_syndrome_aug += 1

    logger.info("证型增强样本: %s", n_syndrome_aug)
    logger.info("总训练样本: %s", len(samples))
    return samples, len(all_symptoms), num_syndromes


def _match_symptoms_in_text_v2(text: str, symptom_list: List[str]) -> List[int]:
    """在文本中匹配症状，支持扩展症状列表"""
    if not text:
        return []
    ids = []
    for i, symptom in enumerate(symptom_list, 1):
        if symptom in text:
            ids.append(i)
    return ids


def _infer_syndrome_v2(
    symptom_ids: List[int],
    symptom_list: List[str],
    syndrome_defs: Dict,
) -> int:
    """根据症状推断证型"""
    symptom_names = {symptom_list[sid - 1] for sid in symptom_ids if 1 <= sid <= len(symptom_list)}
    syndrome_names = list(syndrome_defs.keys())

    best_id = 0
    best_score = 0.0
    for i, (sname, info) in enumerate(syndrome_defs.items(), 1):
        required = info.get("symptoms", [])
        if not required:
            continue
        matched = sum(1 for s in required if s in symptom_names)
        score = matched / max(len(required), 1)
        if score > best_score and score >= 0.4:
            best_score = score
            best_id = i
    return best_id


def build_herb_char_features_v3(data_loader: TCMDataLoader) -> Tuple[torch.Tensor, int, list]:
    """构建药材特征（同 V2，兼容）"""
    herb_texts = {}
    for idx in range(1, data_loader.num_herbs + 1):
        herb_id = data_loader.idx_to_herb_id[idx]
        herb = data_loader.herbs[herb_id]
        text = " ".join([
            herb.get("efficacy", "") or "",
            herb.get("indications", "") or "",
        ])
        herb_texts[idx] = text

    char_set = set()
    for text in herb_texts.values():
        for ch in text:
            if "一" <= ch <= "鿿":
                char_set.add(ch)
    char_list = sorted(char_set)
    logger.info("药材字符词表大小: %s", len(char_list))

    num_herbs = len(herb_texts)
    num_chars = len(char_list)
    char_to_idx = {ch: i for i, ch in enumerate(char_list)}
    multi_hot = np.zeros((num_herbs, num_chars), dtype=np.float32)
    for idx, text in herb_texts.items():
        for ch in text:
            if ch in char_to_idx:
                multi_hot[idx - 1, char_to_idx[ch]] = 1.0

    return torch.from_numpy(multi_hot), num_chars, char_list


# ═══════════════════════════════════════════════════
# 训练器 V3
# ═══════════════════════════════════════════════════

class TrainerV3:
    def __init__(
        self,
        model: TCMRecommenderSystemV3,
        herb_feature_encoder: HerbFeatureEncoderV3,
        herb_char_multihot: torch.Tensor,
        device: str = "cpu",
        lr: float = 0.0005,
        weight_decay: float = 1e-5,
        contrastive_weight: float = 0.15,
    ):
        self.model = model.to(device)
        self.herb_feature_encoder = herb_feature_encoder.to(device)
        self.herb_char_multihot = herb_char_multihot.to(device)
        self.device = device
        self.contrastive_weight = contrastive_weight

        # 损失函数
        self.syndrome_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.formula_criterion = nn.CrossEntropyLoss(ignore_index=0)
        pos_weight = torch.tensor(5.0)
        self.herb_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 优化器
        all_params = list(model.parameters()) + list(herb_feature_encoder.parameters())
        self.optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=8
        )

        self.history = {"train_loss": [], "val_loss": [], "syndrome_acc": [], "herb_map": []}

    def _herb_features_tensor(self) -> torch.Tensor:
        with torch.no_grad():
            return self.herb_feature_encoder(self.herb_char_multihot)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        self.herb_feature_encoder.train()
        total_loss = 0.0
        herb_feats = self._herb_features_tensor()

        for batch in tqdm(dataloader, desc="训练", leave=False):
            symptoms = batch["symptoms"].to(self.device)
            herb_labels = batch["herb_labels"].to(self.device)
            formulas = batch["formula_label"].to(self.device)
            syndromes = batch["syndrome_label"].to(self.device)

            syn_logits, formula_scores, herb_scores, sv, herb_emb = self.model(
                symptoms, herb_feats
            )

            loss = 0.0
            has_loss = False

            # 证型分类损失
            if (syndromes != 0).sum() > 0:
                loss_syn = self.syndrome_criterion(syn_logits, syndromes)
                if not torch.isnan(loss_syn):
                    loss += loss_syn * 0.8
                    has_loss = True

            # 方剂推荐损失
            if (formulas != 0).sum() > 0:
                loss_formula = self.formula_criterion(formula_scores, formulas)
                if not torch.isnan(loss_formula):
                    loss += loss_formula * 0.5
                    has_loss = True

            # 药材多标签损失
            if herb_labels.sum() > 0:
                loss_herb = self.herb_criterion(herb_scores, herb_labels)
                if not torch.isnan(loss_herb):
                    loss += loss_herb * 0.3
                    has_loss = True

            # 对比学习损失
            if herb_labels.sum() > 0:
                loss_cont = self.model.contrastive_loss(sv, herb_emb, herb_labels)
                if not torch.isnan(loss_cont):
                    loss += loss_cont * self.contrastive_weight
                    has_loss = True

            if not has_loss:
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.herb_feature_encoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        self.herb_feature_encoder.eval()
        total_loss = 0.0
        syn_correct = 0
        syn_total = 0
        herb_map_sum = 0.0
        herb_map_count = 0
        herb_feats = self._herb_features_tensor()

        for batch in tqdm(dataloader, desc="验证", leave=False):
            symptoms = batch["symptoms"].to(self.device)
            herb_labels = batch["herb_labels"].to(self.device)
            formulas = batch["formula_label"].to(self.device)
            syndromes = batch["syndrome_label"].to(self.device)

            syn_logits, formula_scores, herb_scores, sv, herb_emb = self.model(
                symptoms, herb_feats
            )

            loss = 0.0
            has_loss = False

            if (syndromes != 0).sum() > 0:
                l = self.syndrome_criterion(syn_logits, syndromes)
                if not torch.isnan(l):
                    loss += l * 0.8
                    has_loss = True

            if (formulas != 0).sum() > 0:
                l = self.formula_criterion(formula_scores, formulas)
                if not torch.isnan(l):
                    loss += l * 0.5
                    has_loss = True

            if herb_labels.sum() > 0:
                l = self.herb_criterion(herb_scores, herb_labels)
                if not torch.isnan(l):
                    loss += l * 0.3
                    has_loss = True

                l_cont = self.model.contrastive_loss(sv, herb_emb, herb_labels)
                if not torch.isnan(l_cont):
                    loss += l_cont * self.contrastive_weight
                    has_loss = True

            if not has_loss:
                continue

            total_loss += loss.item()

            # 证型准确率
            mask = syndromes != 0
            if mask.sum() > 0:
                pred = syn_logits[mask].argmax(dim=1)
                syn_correct += (pred == syndromes[mask]).sum().item()
                syn_total += mask.sum().item()

            # 药材 Mean Average Precision (简化)
            if herb_labels.sum() > 0:
                for i in range(herb_labels.size(0)):
                    if herb_labels[i].sum() == 0:
                        continue
                    _, indices = torch.topk(herb_scores[i], k=min(10, herb_scores.shape[1]))
                    rel = herb_labels[i][indices]
                    ap = 0.0
                    num_hits = 0
                    for k, r in enumerate(rel, 1):
                        if r > 0:
                            num_hits += 1
                            ap += num_hits / k
                    if num_hits > 0:
                        ap /= min(num_hits, herb_labels[i].sum().item())
                    herb_map_sum += ap
                    herb_map_count += 1

        return {
            "loss": total_loss / max(len(dataloader), 1),
            "syndrome_acc": syn_correct / max(syn_total, 1),
            "herb_map": herb_map_sum / max(herb_map_count, 1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 80,
        save_path: str = "best_model_v3.pth",
        encoder_path: str = "best_encoder_v3.pth",
    ):
        best_val_loss = float("inf")
        patience = 0
        early_stop = 20

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["syndrome_acc"].append(val_metrics["syndrome_acc"])
            self.history["herb_map"].append(val_metrics["herb_map"])

            logger.info(
                "Epoch %s/%s  tr=%.4f  val=%.4f  syn_acc=%.3f  herb_map=%.3f",
                epoch + 1, epochs,
                train_loss, val_metrics["loss"],
                val_metrics["syndrome_acc"], val_metrics["herb_map"],
            )
            self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save(save_path, encoder_path)
                logger.info("  -> 保存最佳模型 (val_loss=%.4f)", best_val_loss)
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    logger.info("早停触发 (patience=%s)", early_stop)
                    break

        logger.info("训练完成，最佳 val_loss=%.4f", best_val_loss)

    def _save(self, model_path: str, encoder_path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "model_config": {
                    "d_model": self.model.d_model,
                },
            },
            model_path,
        )
        torch.save(
            {
                "encoder_state_dict": self.herb_feature_encoder.state_dict(),
                "char_multihot": self.herb_char_multihot.cpu(),
            },
            encoder_path,
        )


# ═══════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="中医药推荐系统训练 V3")
    parser.add_argument("--db", type=str, default="data/tcm_knowledge.db")
    parser.add_argument("--output", type=str, default="tcm_recommender/best_model_v3.pth")
    parser.add_argument("--encoder-out", type=str, default="tcm_recommender/best_encoder_v3.pth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    args = parser.parse_args()

    db_path = Path(PROJECT_ROOT) / args.db
    if not db_path.exists():
        logger.error("数据库不存在：%s", db_path)
        return

    # 1. 加载数据
    logger.info("[1/6] 加载数据...")
    data_loader = TCMDataLoader(str(db_path))
    data_loader.load_all_data()

    # 加载扩展配置
    extended_symptoms = None
    extended_syndromes = None
    try:
        from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST, EXTENDED_SYNDROME_DEFINITIONS
        extended_symptoms = EXTENDED_SYMPTOM_LIST
        extended_syndromes = EXTENDED_SYNDROME_DEFINITIONS
        logger.info("已加载扩展配置：%s 症状, %s 证型", len(extended_symptoms), len(extended_syndromes))
    except ImportError:
        logger.warning("扩展配置未找到，使用默认配置")

    # 2. 构建增强训练数据
    logger.info("[2/6] 构建增强训练数据...")
    samples, num_symptoms, num_syndromes = build_enhanced_training_data(
        data_loader, extended_symptoms, extended_syndromes
    )

    # 划分数据集
    indices = np.random.RandomState(42).permutation(len(samples))
    split = int(len(samples) * 0.85)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = TCMDatasetV3([samples[i] for i in train_idx])
    val_ds = TCMDatasetV3([samples[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=False)
    logger.info("训练集: %s, 验证集: %s", len(train_ds), len(val_ds))

    # 3. 构建药材特征
    logger.info("[3/6] 构建药材字符特征...")
    herb_char_multihot, num_chars, char_list = build_herb_char_features_v3(data_loader)
    herb_feature_dim = 128
    herb_feature_encoder = HerbFeatureEncoderV3(num_chars, feature_dim=herb_feature_dim)

    # 4. 创建模型
    logger.info("[4/6] 创建 V3 模型...")
    model = create_model_v3(
        num_symptoms=num_symptoms,
        num_herbs=data_loader.num_herbs,
        num_formulas=len(data_loader.formulas) + 1,
        num_syndromes=num_syndromes + 1,
        herb_char_dim=num_chars,
        d_model=256,
        herb_feature_dim=herb_feature_dim,
        num_transformer_layers=4,
    )

    # 5. 训练
    logger.info("[5/6] 开始训练 V3...")
    trainer = TrainerV3(
        model, herb_feature_encoder, herb_char_multihot,
        device=args.device, lr=args.lr, weight_decay=args.wd,
        contrastive_weight=args.contrastive_weight,
    )
    trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        save_path=args.output,
        encoder_path=args.encoder_out,
    )

    # 6. 测试
    logger.info("[6/6] 测试推荐效果...")
    # 加载最佳模型
    checkpoint = torch.load(args.output, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(args.device)

    if Path(args.encoder_out).exists():
        enc_checkpoint = torch.load(args.encoder_out, map_location=args.device)
        herb_feature_encoder.load_state_dict(enc_checkpoint["encoder_state_dict"])
    herb_feature_encoder.eval()
    herb_feature_encoder.to(args.device)

    test_cases = [
        ([1, 2, 3], "乏力、疲倦、食欲不振"),
        ([5, 6, 9], "头痛、咳嗽、失眠"),
        ([7, 8], "发热、咳嗽"),
        ([10, 11, 12], "腰膝酸软、畏寒、头晕"),
    ]

    herb_names = data_loader.get_herb_names()
    formula_names = data_loader.get_formula_names()
    syn_names = ["未知"] + list(extended_syndromes.keys() if extended_syndromes else [])

    hf = herb_feature_encoder(herb_char_multihot.to(args.device))

    for sym_ids, desc in test_cases:
        test_tensor = torch.tensor([sym_ids], dtype=torch.long).to(args.device)
        result = model.predict(test_tensor, hf, top_k=5)
        print(f"\n{'='*60}")
        print(f"测试: {desc}")
        if result["syndromes"]:
            sid, sp = result["syndromes"][0]
            if sid < len(syn_names):
                print(f"证型: {syn_names[sid]} ({sp:.3f})")
        print("药材 Top-5:")
        for idx, score in result["herbs"]:
            print(f"  {herb_names.get(idx, '?'):12s} {score:.4f}")
        print("方剂 Top-5:")
        for idx, score in result["formulas"]:
            print(f"  {formula_names.get(idx, '?'):16s} {score:.4f}")

    # 安全校验测试
    checker = SafetyChecker(data_loader.incompatible_herbs)
    top_h_names = [herb_names.get(i, "") for i, _ in result["herbs"][:5]]
    safety = checker.check_compatibility(top_h_names)
    if safety["warnings"]:
        print("\n安全提示:")
        for w in safety["warnings"]:
            print(f"  {w}")
    else:
        print("\n安全提示: [OK]")

    data_loader.close()
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
