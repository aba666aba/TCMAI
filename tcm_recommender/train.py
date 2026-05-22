# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 训练脚本

管线：数据加载 -> 特征构建 -> 模型训练 -> 评估保存
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_pipeline import TCMDataPipeline
from tcm_recommender.model import TCMNet, create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────
MAX_SYMPTOMS = 32
NATURE_LIST = ["寒", "热", "温", "凉", "平"]
FLAVOR_LIST = ["酸", "苦", "甘", "辛", "咸"]


# ═══════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════

class TCMDataset(Dataset):
    """训练数据集"""

    def __init__(self, data: Dict[str, np.ndarray], indices: Optional[np.ndarray] = None):
        if indices is not None:
            self.symptom_ids = data["symptom_ids"][indices]
            self.symptom_mask = data["symptom_mask"][indices]
            self.herb_labels = data["herb_labels"][indices]
            self.formula_labels = data["formula_labels"][indices]
            self.syndrome_labels = data["syndrome_labels"][indices]
        else:
            self.symptom_ids = data["symptom_ids"]
            self.symptom_mask = data["symptom_mask"]
            self.herb_labels = data["herb_labels"]
            self.formula_labels = data["formula_labels"]
            self.syndrome_labels = data["syndrome_labels"]

    def __len__(self):
        return len(self.symptom_ids)

    def __getitem__(self, idx):
        return {
            "symptom_ids": torch.from_numpy(self.symptom_ids[idx].astype(np.int64)),
            "symptom_mask": torch.from_numpy(self.symptom_mask[idx].astype(np.float32)),
            "herb_labels": torch.from_numpy(self.herb_labels[idx].astype(np.float32)),
            "formula_label": int(self.formula_labels[idx]),
            "syndrome_label": int(self.syndrome_labels[idx]),
        }


# ═══════════════════════════════════════════════════════════
# 药材特征构建
# ═══════════════════════════════════════════════════════════

def build_herb_features(pipeline: TCMDataPipeline) -> Dict[str, torch.Tensor]:
    """构建药材的字符多热矩阵 + 类别/性味/归经索引"""
    nh = pipeline.num_herbs

    # ── 字符多热矩阵 ──
    herb_texts = {}
    for model_idx in range(1, nh + 1):
        db_id = pipeline.model_idx_to_herb_db_id[model_idx]
        herb = pipeline.herbs[db_id]
        herb_texts[model_idx] = f"{herb.get('efficacy', '')} {herb.get('indications', '')}"

    char_set = set()
    for text in herb_texts.values():
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                char_set.add(ch)
    char_list = sorted(char_set)
    char_to_idx = {ch: i for i, ch in enumerate(char_list)}
    num_chars = len(char_list)

    char_mh = torch.zeros(nh, num_chars)
    for mi, text in herb_texts.items():
        for ch in text:
            if ch in char_to_idx:
                char_mh[mi - 1, char_to_idx[ch]] = 1.0

    # ── 类别索引 ──
    # 收集所有类别名
    all_categories = sorted(set(
        pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("category", "") or "未知"
        for mi in range(1, nh + 1)
    ))
    cat_to_idx = {c: i + 1 for i, c in enumerate(all_categories)}
    cat_idx = torch.zeros(nh, dtype=torch.long)
    for mi in range(1, nh + 1):
        cat = pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("category", "") or "未知"
        cat_idx[mi - 1] = cat_to_idx.get(cat, 0)

    # ── 药性索引 ──
    nat_idx = torch.zeros(nh, dtype=torch.long)
    for mi in range(1, nh + 1):
        nat = pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("nature", "") or ""
        found = False
        for ni, n_name in enumerate(NATURE_LIST, 1):
            if n_name in nat:
                nat_idx[mi - 1] = ni
                found = True
                break
        if not found:
            nat_idx[mi - 1] = len(NATURE_LIST) + 1  # unknown category

    # ── 药味索引 (取第一个) ──
    flv_idx = torch.zeros(nh, dtype=torch.long)
    for mi in range(1, nh + 1):
        flv = pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("flavor", "") or ""
        found = False
        for fi, f_name in enumerate(FLAVOR_LIST, 1):
            if f_name in flv:
                flv_idx[mi - 1] = fi
                found = True
                break
        if not found:
            flv_idx[mi - 1] = len(FLAVOR_LIST) + 1  # unknown category

    logger.info("药材特征: chars=%s, categories=%s", num_chars, len(all_categories))
    return {
        "char_multihot": char_mh,
        "category_idx": cat_idx,
        "nature_idx": nat_idx,
        "flavor_idx": flv_idx,
        "num_chars": num_chars,
        "num_categories": len(all_categories) + 1,  # +1 for padding
        "num_natures": len(NATURE_LIST) + 2,  # +1 padding +1 unknown
        "num_flavors": len(FLAVOR_LIST) + 2,
    }


# ═══════════════════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model: TCMNet, herb_features: Dict[str, torch.Tensor],
                 device: torch.device, lr: float = 1e-4, weight_decay: float = 1e-5,
                 warmup_steps: int = 500, grad_accum: int = 1,
                 contrastive_weight: float = 0.2, label_smoothing: float = 0.05):
        self.model = model.to(device)
        self.device = device
        self.grad_accum = grad_accum
        self.contrastive_weight = contrastive_weight
        self.label_smoothing = label_smoothing

        # 药材特征预加载到设备
        self.herb_char = herb_features["char_multihot"].to(device)
        self.herb_cat = herb_features["category_idx"].to(device)
        self.herb_nat = herb_features["nature_idx"].to(device)
        self.herb_flv = herb_features["flavor_idx"].to(device)

        # 损失函数
        self.syn_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
        self.formula_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

        # 优化器
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
        optimizer_grouped_params = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=lr, betas=(0.9, 0.98))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01)

        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        self.history = {"train": [], "val": [], "lr": []}
        self.global_step = 0

    def _herb_bce_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """加权 BCE 损失（处理正负样本不平衡）"""
        pos_mask = labels > 0.5
        neg_mask = labels < -0.5
        normal_mask = ~pos_mask & ~neg_mask  # 0标签（忽略的）

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # 只计算正类和部分负类
        pos_weight_val = (neg_mask.float().sum() / pos_mask.float().sum().clamp(min=1)).clamp(1, 50)
        weight = torch.ones_like(labels)
        weight[pos_mask] = pos_weight_val

        loss = F.binary_cross_entropy_with_logits(
            scores[normal_mask | pos_mask],
            (labels > 0.5).float()[normal_mask | pos_mask],
            weight=weight[normal_mask | pos_mask],
        )
        return loss

    def train_step(self, batch: dict) -> Dict[str, float]:
        self.model.train()

        sym_ids = batch["symptom_ids"].to(self.device)
        sym_mask = batch["symptom_mask"].to(self.device)
        herb_lbl = batch["herb_labels"].to(self.device)
        form_lbl = batch["formula_label"].to(self.device)
        syn_lbl = batch["syndrome_label"].to(self.device)

        # 混合精度 (仅 CUDA)
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            syn_logits, herb_scores, form_scores, sv, hv = self.model(
                sym_ids, sym_mask, self.herb_char, self.herb_cat, self.herb_nat, self.herb_flv)

            loss = 0.0
            metrics = {}

            # 证型分类
            mask_syn = syn_lbl != 0
            if mask_syn.sum() > 0:
                l_syn = self.syn_criterion(syn_logits, syn_lbl)
                if not torch.isnan(l_syn):
                    loss += l_syn
                    metrics["syn_loss"] = l_syn.item()

            # 方剂排序
            mask_form = form_lbl > 0
            if mask_form.sum() > 0:
                l_form = self.formula_criterion(form_scores[mask_form], form_lbl[mask_form])
            else:
                l_form = self.formula_criterion(form_scores, form_lbl)
            if not torch.isnan(l_form):
                loss += l_form * 0.5
                metrics["form_loss"] = l_form.item()

            # 药材多标签
            has_herb = (herb_lbl.abs() > 0.5).sum() > 0
            if has_herb:
                l_herb = self._herb_bce_loss(herb_scores, herb_lbl)
                if not torch.isnan(l_herb) and l_herb.item() > 0:
                    loss += l_herb * 0.3
                    metrics["herb_loss"] = l_herb.item()

            # 对比学习
            l_cont = self.model.contrastive_loss(sv, hv, herb_lbl)
            if not torch.isnan(l_cont):
                loss += l_cont * self.contrastive_weight
                metrics["cont_loss"] = l_cont.item()

        # 反向传播
        loss = loss / self.grad_accum
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics["total_loss"] = loss.item() * self.grad_accum
        return metrics

    def optimizer_step(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_step += 1

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_syn_loss = total_herb_loss = total_form_loss = 0.0
        syn_correct = syn_total = 0
        herb_hits10 = herb_total_pos = 0
        n_batches = 0

        for batch in tqdm(dataloader, desc="Eval", leave=False):
            sym_ids = batch["symptom_ids"].to(self.device)
            sym_mask = batch["symptom_mask"].to(self.device)
            herb_lbl = batch["herb_labels"].to(self.device)
            form_lbl = batch["formula_label"].to(self.device)
            syn_lbl = batch["syndrome_label"].to(self.device)

            syn_logits, herb_scores, form_scores, sv, hv = self.model(
                sym_ids, sym_mask, self.herb_char, self.herb_cat, self.herb_nat, self.herb_flv)

            # 证型准确率
            mask_syn = syn_lbl != 0
            if mask_syn.sum() > 0:
                pred = syn_logits[mask_syn].argmax(dim=1)
                syn_correct += (pred == syn_lbl[mask_syn]).sum().item()
                syn_total += mask_syn.sum().item()
                l_syn = self.syn_criterion(syn_logits, syn_lbl)
                if not torch.isnan(l_syn):
                    total_syn_loss += l_syn.item()

            # 药材 Hit@10
            has_herb = (herb_lbl.abs() > 0.5).sum(dim=1) > 0
            if has_herb.any():
                l_h = self._herb_bce_loss(herb_scores, herb_lbl)
                if not torch.isnan(l_h):
                    total_herb_loss += l_h.item()
                for i in range(herb_scores.size(0)):
                    if not has_herb[i]:
                        continue
                    pos_mask = herb_lbl[i] > 0.5
                    if pos_mask.sum() == 0:
                        continue
                    _, topk = torch.topk(herb_scores[i], k=min(10, herb_scores.shape[1]))
                    herb_hits10 += sum(1 for idx in topk.tolist() if pos_mask[idx].item())
                    herb_total_pos += pos_mask.sum().item()

            # 方剂
            mask_form = form_lbl > 0
            if mask_form.sum() > 0:
                l_f = self.formula_criterion(form_scores[mask_form], form_lbl[mask_form])
            else:
                l_f = self.formula_criterion(form_scores, form_lbl)
            if not torch.isnan(l_f):
                total_form_loss += l_f.item()

            n_batches += 1

        n = max(n_batches, 1)
        return {
            "val_loss": (total_syn_loss + total_herb_loss * 0.3 + total_form_loss * 0.5) / n,
            "syn_acc": syn_correct / max(syn_total, 1),
            "herb_recall10": herb_hits10 / max(herb_total_pos, 1),
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, save_dir: str = "tcm_recommender",
              save_prefix: str = "best_model"):
        best_val_loss = float("inf")
        patience = 0
        early_stop_patience = 15
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_metrics = {"syn_loss": 0, "form_loss": 0, "herb_loss": 0, "cont_loss": 0, "total_loss": 0}
            n_steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                n_steps += 1

                if (batch_idx + 1) % self.grad_accum == 0:
                    self.optimizer_step()

                pbar.set_postfix({k: f"{v / n_steps:.3f}" for k, v in epoch_metrics.items()
                                  if k in ("total_loss", "syn_loss")})

            # 处理最后的不完整梯度累积
            if (batch_idx + 1) % self.grad_accum != 0:
                self.optimizer_step()

            # 验证
            val_metrics = self.evaluate(val_loader)
            epoch_time = time.time() - epoch_start

            logger.info(
                "Epoch %3d | tr_loss=%.4f | val=%.4f syn_acc=%.3f herb_r@10=%.3f | time=%.1fs lr=%.2e",
                epoch + 1, epoch_metrics["total_loss"] / n_steps,
                val_metrics["val_loss"], val_metrics["syn_acc"],
                val_metrics["herb_recall10"], epoch_time,
                self.optimizer.param_groups[0]["lr"],
            )

            # 保存
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self._save(f"{save_dir}/{save_prefix}.pth", epoch, val_metrics)
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    logger.info("Early stopping at epoch %s", epoch + 1)
                    break

            self.history["train"].append(epoch_metrics["total_loss"] / n_steps)
            self.history["val"].append(val_metrics["val_loss"])

        logger.info("训练完成, best_val=%.4f", best_val_loss)

    def _save(self, path: str, epoch: int, metrics: dict):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "history": self.history,
            "config": {
                "d_model": self.model.d_model,
                "num_layers": len([m for m in self.model.symptom_encoder.layers]),
                "version": "v2",
            },
        }, path)
        logger.info("模型已保存: %s", path)


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TCM 训练")
    parser.add_argument("--db", default="data/tcm_knowledge.db")
    parser.add_argument("--data", default="data/train_data.npz")
    parser.add_argument("--output", default="tcm_recommender/best_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=100000,
                        help="生成的训练样本数（如果 npz 不存在则生成）")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=384,
                        help="模型维度 (512 for 8M params, 384 for ~5M)")
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("设备: %s", device)

    # ═══ 1. 数据 ═══
    data_path = Path(args.data)
    if not data_path.exists():
        logger.info("生成训练数据 (%s 样本)...", args.samples)
        pipeline = TCMDataPipeline(args.db)
        pipeline.load_all()
        data = pipeline.generate_samples(args.samples)
        pipeline.save(data, str(data_path))
        pipeline.close()
    else:
        logger.info("加载已有数据: %s", data_path)
        data = dict(np.load(data_path, allow_pickle=True))

    n_samples = int(data["num_samples"])
    n_symptoms = int(data["num_symptoms"])
    n_herbs = int(data["num_herbs"])
    n_formulas = int(data["num_formulas"])
    n_syndromes = int(data["num_syndromes"])
    logger.info("数据: %s 样本, %s 症状, %s 药材, %s 方剂, %s 证型",
                 n_samples, n_symptoms, n_herbs, n_formulas, n_syndromes)

    # 划分
    dataset = TCMDataset(data)
    n_train = int(n_samples * 0.85)
    n_val = n_samples - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    logger.info("训练集: %s | 验证集: %s", n_train, n_val)

    # ═══ 2. 特征 ═══
    logger.info("构建药材特征...")
    pipeline = TCMDataPipeline(args.db)
    pipeline.load_all()
    herb_feat = build_herb_features(pipeline)
    pipeline.close()

    # ═══ 3. 模型 ═══
    logger.info("创建模型...")
    model = create_model(
        num_symptoms=n_symptoms,
        num_herbs=n_herbs,
        num_formulas=n_formulas + 1,
        num_syndromes=n_syndromes + 1,
        num_chars=herb_feat["num_chars"],
        num_categories=herb_feat["num_categories"],
        d_model=args.d_model,
        nhead=8,
        num_layers=args.num_layers,
        dropout=0.1,
    )

    # ═══ 4. 训练 ═══
    trainer = Trainer(
        model, herb_feat, device,
        lr=args.lr, grad_accum=args.grad_accum,
        contrastive_weight=args.contrastive_weight,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs, save_prefix="best_model")

    # ═══ 5. 最终测试 ═══
    logger.info("最终测试...")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # 简单的推理测试
    val_metrics = trainer.evaluate(val_loader)
    logger.info("最终验证: loss=%.4f syn_acc=%.3f herb_r@10=%.3f",
                 val_metrics["val_loss"], val_metrics["syn_acc"], val_metrics["herb_recall10"])

    # 保存最终结果
    results_path = Path(args.output).with_suffix(".json")
    results_path.write_text(json.dumps({
        "val_metrics": {k: float(v) if isinstance(v, (np.floating, float, np.integer, int))
                        else v for k, v in val_metrics.items()},
        "config": vars(args),
    }, indent=2, ensure_ascii=False))

    logger.info("全部完成!")


if __name__ == "__main__":
    main()
