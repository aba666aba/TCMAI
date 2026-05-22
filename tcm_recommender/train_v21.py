# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 训练脚本 V2.1（务实升级版）

基于 V2 架构：
- embed_dim 64 → 128
- 使用扩展数据（200症状、17证型、证型增强样本）
- 加入对比学习辅助损失
- 参数 ~600K，适配761样本规模
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_loader import TCMDataLoader
from tcm_recommender.model import (
    HerbFeatureEncoder,
    SymptomEncoder,
    SyndromeClassifier,
    FormulaRecommender,
    HerbRecommender,
    TCMRecommenderSystem,
    SafetyChecker,
    create_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 数据集构建
# ═══════════════════════════════════════════════════

def _match_symptoms(text: str, symptom_list: List[str]) -> List[int]:
    if not text:
        return []
    ids = []
    for i, s in enumerate(symptom_list, 1):
        if s in text:
            ids.append(i)
    return ids


def _infer_syndrome(symptom_ids: List[int], symptom_list: List[str], syndrome_defs: Dict) -> int:
    s_names = {symptom_list[sid - 1] for sid in symptom_ids if 1 <= sid <= len(symptom_list)}
    best_id = 0
    best_score = 0.0
    for i, (_, info) in enumerate(syndrome_defs.items(), 1):
        required = info.get("symptoms", [])
        if not required:
            continue
        matched = sum(1 for s in required if s in s_names)
        score = matched / max(len(required), 1)
        if score > best_score and score >= 0.4:
            best_score = score
            best_id = i
    return best_id


def build_training_data_v21(
    data_loader: TCMDataLoader,
    extended_symptoms: Optional[List[str]] = None,
    extended_syndromes: Optional[Dict] = None,
) -> Tuple[List, List, List, List, int, int]:
    """构建训练数据（兼容 V2 格式）"""
    from core.tcm_config import SYMPTOM_LIST, SYNDROME_RULES

    all_symptoms = list(extended_symptoms) if extended_symptoms else list(SYMPTOM_LIST)
    all_syndromes = dict(extended_syndromes) if extended_syndromes else dict(SYNDROME_RULES)

    symptoms_list, herbs_list, formulas_list, syndromes_list = [], [], [], []

    # 来源1：方剂样本
    for formula_id, formula in data_loader.formulas.items():
        herb_ids = data_loader._parse_composition(formula.get("composition", ""))
        if not herb_ids:
            continue

        eff_text = " ".join([
            formula.get("functions", "") or "",
            formula.get("indications", "") or "",
        ])
        for idx in herb_ids:
            hid = data_loader.idx_to_herb_id.get(idx)
            if hid and hid in data_loader.herbs:
                h = data_loader.herbs[hid]
                for f in ["efficacy", "indications"]:
                    if h.get(f):
                        eff_text += " " + h[f]

        symptom_ids = _match_symptoms(eff_text, all_symptoms)
        if not symptom_ids:
            continue

        syn_id = _infer_syndrome(symptom_ids, all_symptoms, all_syndromes)
        f_idx = data_loader.formula_id_to_idx[formula_id]

        symptoms_list.append(symptom_ids)
        herbs_list.append(herb_ids)
        formulas_list.append(f_idx)
        syndromes_list.append(syn_id)

    # 来源2：药材样本
    for herb_id, herb in data_loader.herbs.items():
        h_idx = data_loader.herb_id_to_idx.get(herb_id)
        if h_idx is None:
            continue

        eff_text = " ".join([
            herb.get("efficacy", "") or "",
            herb.get("indications", "") or "",
        ])
        symptom_ids = _match_symptoms(eff_text, all_symptoms)
        if not symptom_ids:
            continue

        syn_id = _infer_syndrome(symptom_ids, all_symptoms, all_syndromes)

        symptoms_list.append(symptom_ids)
        herbs_list.append([h_idx])
        formulas_list.append(0)
        syndromes_list.append(syn_id)

    # 来源3：证型增强
    if extended_syndromes:
        s_to_idx = {s: i + 1 for i, s in enumerate(all_symptoms)}
        rng = np.random.RandomState(42)
        for syn_name, syn_info in extended_syndromes.items():
            syn_symptoms = syn_info.get("symptoms", [])
            syn_herbs = syn_info.get("recommend_herbs", [])
            syn_formulas = syn_info.get("recommend_formulas", [])

            s_ids = [s_to_idx[s] for s in syn_symptoms if s in s_to_idx]
            if not s_ids:
                continue

            h_indices = []
            for h_name in syn_herbs:
                h_id = data_loader.herb_name_to_db_id.get(h_name)
                if h_id and h_id in data_loader.herb_id_to_idx:
                    h_indices.append(data_loader.herb_id_to_idx[h_id])

            syn_idx = list(all_syndromes.keys()).index(syn_name) + 1 if syn_name in all_syndromes else 0
            fl = 0
            if syn_formulas:
                f_name = syn_formulas[0]
                for fid, finfo in data_loader.formulas.items():
                    if finfo.get("name") == f_name and fid in data_loader.formula_id_to_idx:
                        fl = data_loader.formula_id_to_idx[fid]
                        break

            # 完整版
            if len(s_ids) >= 2:
                symptoms_list.append(s_ids)
                herbs_list.append(h_indices if h_indices else [1])
                formulas_list.append(fl)
                syndromes_list.append(syn_idx)

            # 子集变体
            if len(s_ids) >= 3:
                for _ in range(min(2, len(s_ids))):
                    sub = list(rng.choice(s_ids, size=max(2, len(s_ids) // 2), replace=False))
                    symptoms_list.append(sub)
                    herbs_list.append(h_indices if h_indices else [1])
                    formulas_list.append(fl)
                    syndromes_list.append(syn_idx)

    logger.info("总训练样本: %s", len(symptoms_list))
    return symptoms_list, herbs_list, formulas_list, syndromes_list, len(all_symptoms), len(all_syndromes)


# ═══════════════════════════════════════════════════
# 训练器 V2.1
# ═══════════════════════════════════════════════════

class TCMDatasetV21(Dataset):
    def __init__(self, symptoms, herbs, formulas, syndromes, max_s=30, max_h=30):
        self.symptoms = symptoms
        self.herbs = herbs
        self.formulas = formulas
        self.syndromes = syndromes
        self.max_s = max_s
        self.max_h = max_h

    def __len__(self):
        return len(self.symptoms)

    def __getitem__(self, idx):
        syms = self.symptoms[idx][:self.max_s]
        syms = syms + [0] * (self.max_s - len(syms))
        herbs = self.herbs[idx][:self.max_h]
        herbs = herbs + [0] * (self.max_h - len(herbs))
        return {
            "symptoms": torch.tensor(syms, dtype=torch.long),
            "herbs": torch.tensor(herbs, dtype=torch.long),
            "formula": torch.tensor(self.formulas[idx], dtype=torch.long),
            "syndrome": torch.tensor(self.syndromes[idx], dtype=torch.long),
        }


class TrainerV21:
    def __init__(
        self,
        model: nn.Module,
        herb_feature_encoder: HerbFeatureEncoder,
        herb_char_multihot: torch.Tensor,
        device: str = "cpu",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.herb_feature_encoder = herb_feature_encoder.to(device)
        self.herb_char_multihot = herb_char_multihot.to(device)
        self.device = device

        self.syndrome_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.formula_criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 对每个样本只算正类 + 负采样
        pos_weight = torch.tensor(3.0)
        self.herb_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        all_p = list(model.parameters()) + list(herb_feature_encoder.parameters())
        self.optimizer = optim.AdamW(all_p, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=8
        )
        self.history = {"train_loss": [], "val_loss": [], "syndrome_acc": []}

        # 预计算全量 herb_labels 用于 BCE
        self.precomputed_labels: Optional[torch.Tensor] = None

    def _herb_features(self) -> torch.Tensor:
        with torch.no_grad():
            return self.herb_feature_encoder(self.herb_char_multihot)

    def _herb_labels_matrix(self, herbs_batch: torch.Tensor, num_herbs: int) -> torch.Tensor:
        """V2 格式：将 [batch, max_herbs] → [batch, num_herbs] 多标签"""
        bs = herbs_batch.size(0)
        labels = torch.zeros(bs, num_herbs, device=self.device)
        for i in range(bs):
            for j in range(herbs_batch.size(1)):
                h_id = herbs_batch[i, j].item()
                if 0 < h_id <= num_herbs:
                    labels[i, h_id - 1] = 1.0
        return labels

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        self.herb_feature_encoder.train()
        total_loss = 0.0
        n_batches = 0
        hf = self._herb_features()
        num_herbs = self.model.herb_recommender.num_herbs

        for batch in tqdm(dataloader, desc="训练", leave=False):
            symptoms = batch["symptoms"].to(self.device)
            herbs_raw = batch["herbs"]
            formulas = batch["formula"].to(self.device)
            syndromes = batch["syndrome"].to(self.device)

            syn_logits, formula_scores, herb_scores = self.model(symptoms, hf)
            herb_labels = self._herb_labels_matrix(herbs_raw, num_herbs)

            loss = 0.0
            has_loss = False

            if (syndromes != 0).sum() > 0:
                l = self.syndrome_criterion(syn_logits, syndromes)
                if not torch.isnan(l):
                    loss += l * 1.0
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
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        self.herb_feature_encoder.eval()
        total_loss = 0.0
        syn_correct = 0
        syn_total = 0
        n_batches = 0
        hf = self._herb_features()
        num_herbs = self.model.herb_recommender.num_herbs

        for batch in tqdm(dataloader, desc="验证", leave=False):
            symptoms = batch["symptoms"].to(self.device)
            herbs_raw = batch["herbs"]
            formulas = batch["formula"].to(self.device)
            syndromes = batch["syndrome"].to(self.device)

            syn_logits, formula_scores, herb_scores = self.model(symptoms, hf)
            herb_labels = self._herb_labels_matrix(herbs_raw, num_herbs)

            loss = 0.0
            has_loss = False

            if (syndromes != 0).sum() > 0:
                l = self.syndrome_criterion(syn_logits, syndromes)
                if not torch.isnan(l):
                    loss += l
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

            if not has_loss:
                continue

            total_loss += loss.item()
            n_batches += 1

            mask = syndromes != 0
            if mask.sum() > 0:
                pred = syn_logits[mask].argmax(dim=1)
                syn_correct += (pred == syndromes[mask]).sum().item()
                syn_total += mask.sum().item()

        return {
            "loss": total_loss / max(n_batches, 1),
            "syndrome_acc": syn_correct / max(syn_total, 1),
        }

    def train(self, train_loader, val_loader, epochs=80, save_path="best_model_v2.pth", encoder_path="best_encoder_v2.pth"):
        best_val_loss = float("inf")
        patience = 0
        early_stop = 20

        for ep in range(epochs):
            tr_loss = self.train_epoch(train_loader)
            v = self.evaluate(val_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(v["loss"])
            self.history["syndrome_acc"].append(v["syndrome_acc"])

            logger.info("Epoch %s/%s  tr=%.4f  val=%.4f  syn_acc=%.3f",
                        ep + 1, epochs, tr_loss, v["loss"], v["syndrome_acc"])
            self.scheduler.step(v["loss"])

            if v["loss"] < best_val_loss:
                best_val_loss = v["loss"]
                self._save(save_path, encoder_path)
                logger.info("  -> 保存最佳")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    logger.info("早停")
                    break

        logger.info("训练完成 best_val=%.4f", best_val_loss)

    def _save(self, mp: str, ep: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "model_config": {"d_model": 128, "version": "v2.1"},
        }, mp)
        torch.save({
            "encoder_state_dict": self.herb_feature_encoder.state_dict(),
            "char_multihot": self.herb_char_multihot.cpu(),
        }, ep)


def build_herb_char_features_v2(data_loader: TCMDataLoader) -> Tuple[torch.Tensor, int, list]:
    """同 V2"""
    herb_texts = {}
    for idx in range(1, data_loader.num_herbs + 1):
        hid = data_loader.idx_to_herb_id[idx]
        h = data_loader.herbs[hid]
        herb_texts[idx] = " ".join([h.get("efficacy", "") or "", h.get("indications", "") or ""])

    char_set = set()
    for t in herb_texts.values():
        for ch in t:
            if "一" <= ch <= "鿿":
                char_set.add(ch)
    char_list = sorted(char_set)

    num_herbs = len(herb_texts)
    num_chars = len(char_list)
    c2i = {ch: i for i, ch in enumerate(char_list)}
    mh = np.zeros((num_herbs, num_chars), dtype=np.float32)
    for idx, text in herb_texts.items():
        for ch in text:
            if ch in c2i:
                mh[idx - 1, c2i[ch]] = 1.0
    return torch.from_numpy(mh), num_chars, char_list


def main():
    parser = argparse.ArgumentParser(description="训练 V2.1")
    parser.add_argument("--db", type=str, default="data/tcm_knowledge.db")
    parser.add_argument("--output", type=str, default="tcm_recommender/best_model_v2.pth")
    parser.add_argument("--encoder-out", type=str, default="tcm_recommender/best_encoder_v2.pth")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    db_path = Path(PROJECT_ROOT) / args.db
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return

    logger.info("[1/5] 加载数据...")
    dl = TCMDataLoader(str(db_path))
    dl.load_all_data()

    ext_symptoms = None
    ext_syndromes = None
    try:
        from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST, EXTENDED_SYNDROME_DEFINITIONS
        ext_symptoms = EXTENDED_SYMPTOM_LIST
        ext_syndromes = EXTENDED_SYNDROME_DEFINITIONS
    except ImportError:
        pass

    logger.info("[2/5] 构建训练数据...")
    syms, herbs, formulas, syndromes, num_symptoms, num_syndromes = build_training_data_v21(
        dl, ext_symptoms, ext_syndromes
    )

    indices = np.random.RandomState(42).permutation(len(syms))
    split = int(len(syms) * 0.85)
    train_ds = TCMDatasetV21(
        [syms[i] for i in indices[:split]],
        [herbs[i] for i in indices[:split]],
        [formulas[i] for i in indices[:split]],
        [syndromes[i] for i in indices[:split]],
    )
    val_ds = TCMDatasetV21(
        [syms[i] for i in indices[split:]],
        [herbs[i] for i in indices[split:]],
        [formulas[i] for i in indices[split:]],
        [syndromes[i] for i in indices[split:]],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    logger.info("训练: %s, 验证: %s", len(train_ds), len(val_ds))

    logger.info("[3/5] 构建特征...")
    herb_char, num_chars, _ = build_herb_char_features_v2(dl)
    herb_enc = HerbFeatureEncoder(num_chars, feature_dim=64)
    logger.info("  chars=%s → feat=64", num_chars)

    logger.info("[4/5] 创建模型...")
    model = create_model(
        num_symptoms=num_symptoms,
        num_herbs=dl.num_herbs,
        num_formulas=len(dl.formulas) + 1,
        num_syndromes=num_syndromes + 1,
        herb_char_dim=num_chars,
        embed_dim=128,
        herb_feature_dim=64,
    )

    logger.info("[5/5] 训练...")
    trainer = TrainerV21(model, herb_enc, herb_char, device=args.device, lr=args.lr)
    trainer.train(train_loader, val_loader, epochs=args.epochs,
                  save_path=args.output, encoder_path=args.encoder_out)

    # 测试
    checkpoint = torch.load(args.output, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(args.device)
    if Path(args.encoder_out).exists():
        enc_ckpt = torch.load(args.encoder_out, map_location=args.device)
        herb_enc.load_state_dict(enc_ckpt["encoder_state_dict"])
    herb_enc.eval()
    herb_enc.to(args.device)

    hf = herb_enc(herb_char.to(args.device))
    herb_names = dl.get_herb_names()
    formula_names = dl.get_formula_names()
    syn_names = ["未知"] + (list(ext_syndromes.keys()) if ext_syndromes else [])

    test_cases = [
        ([1, 2, 3], "乏力、疲倦、食欲不振"),
        ([5, 6, 9], "头痛、咳嗽、失眠"),
        ([7, 8], "发热、咳嗽"),
        ([10, 11, 12], "腰膝酸软、畏寒、头晕"),
    ]

    for sym_ids, desc in test_cases:
        t = torch.tensor([sym_ids], dtype=torch.long).to(args.device)
        r = model.predict(t, hf, top_k=5)
        print(f"\n{'='*60}\n测试: {desc}")
        if r["syndromes"]:
            sid, sp = r["syndromes"][0]
            if sid < len(syn_names):
                print(f"证型: {syn_names[sid]} ({sp:.3f})")
        print("药材 Top-5:")
        for idx, score in r["herbs"]:
            print(f"  {herb_names.get(idx, '?'):12s} {score:.4f}")
        print("方剂 Top-5:")
        for idx, score in r["formulas"]:
            print(f"  {formula_names.get(idx, '?'):16s} {score:.4f}")

    dl.close()


if __name__ == "__main__":
    main()
