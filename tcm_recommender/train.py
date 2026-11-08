# Copyright (C) 2026 aba666aba
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

"""训练脚本 — 数据加载 -> 特征构建 -> 模型训练 -> 评估保存"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_pipeline import TCMDataPipeline  # noqa: E402
from tcm_recommender.extended_config import is_toxic_herb  # noqa: E402
from tcm_recommender.model import TCMNet, create_model  # noqa: E402


def _to_tensor(arr, dtype=None):
    """numpy→tensor 的 ABI 安全转换（走 frombuffer 原始内存，绕开 numpy 桥接）。

    部分云镜像的 torch 与环境 numpy 编译 ABI 不匹配（from_numpy 抛
    "expected np.ndarray"），frombuffer 直接消费缓冲区，不经过
    torch 的 numpy 类型检查，任何环境都可用且比 tolist 回退快数十倍。
    """
    if dtype is None:
        dtype = torch.long if np.asarray(arr).dtype.kind in "iu" else torch.float32
    np_dtype = {torch.float32: np.float32, torch.long: np.int64,
                torch.float64: np.float64, torch.int32: np.int32}[dtype]
    a = np.ascontiguousarray(arr, dtype=np_dtype)
    t = torch.frombuffer(bytearray(a.tobytes()), dtype=dtype)
    return t.reshape(a.shape)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 常量
MAX_SYMPTOMS = 32
NATURE_LIST = ["寒", "热", "温", "凉", "平"]
FLAVOR_LIST = ["酸", "苦", "甘", "辛", "咸"]
MERIDIAN_LIST = ["心", "肝", "脾", "肺", "肾", "胆", "胃",
                 "大肠", "小肠", "膀胱", "三焦", "心包"]


# 数据集


class TCMDataset(Dataset):
    """训练数据集"""

    def __init__(
        self, data: dict[str, np.ndarray], indices: Optional[np.ndarray] = None
    ):
        if indices is not None:
            self.symptom_ids = data["symptom_ids"][indices]
            self.symptom_mask = data["symptom_mask"][indices]
            self.herb_labels = data["herb_labels"][indices]
            self.formula_labels = data["formula_labels"][indices]
            self.syndrome_labels = data["syndrome_labels"][indices]
            self.sample_weights = data.get(
                "sample_weights", np.ones(len(indices), dtype=np.float32)
            )[indices]
        else:
            self.symptom_ids = data["symptom_ids"]
            self.symptom_mask = data["symptom_mask"]
            self.herb_labels = data["herb_labels"]
            self.formula_labels = data["formula_labels"]
            self.syndrome_labels = data["syndrome_labels"]
            self.sample_weights = data.get(
                "sample_weights", np.ones(len(self.symptom_ids), dtype=np.float32)
            )

    def __len__(self):
        return len(self.symptom_ids)

    def __getitem__(self, idx):
        return {
            "symptom_ids": _to_tensor(self.symptom_ids[idx].astype(np.int64), torch.long),
            "symptom_mask": _to_tensor(self.symptom_mask[idx].astype(np.float32), torch.float32),
            "herb_labels": _to_tensor(self.herb_labels[idx].astype(np.float32), torch.float32),
            "formula_label": int(self.formula_labels[idx]),
            "syndrome_label": int(self.syndrome_labels[idx]),
            "sample_weight": float(self.sample_weights[idx]),
        }


# 药材特征构建


def build_herb_features(pipeline: TCMDataPipeline) -> dict[str, torch.Tensor]:
    """构建药材的字符多热矩阵 + 类别/性味/归经索引"""
    nh = pipeline.num_herbs

    # ── 字符多热矩阵 ──
    herb_texts = {}
    for model_idx in range(1, nh + 1):
        db_id = pipeline.model_idx_to_herb_db_id[model_idx]
        herb = pipeline.herbs[db_id]
        herb_texts[model_idx] = (
            f"{herb.get('efficacy', '')} {herb.get('indications', '')}"
        )

    char_set = set()
    for text in herb_texts.values():
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
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
    all_categories = sorted(
        {
            pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("category", "")
            or "未知"
            for mi in range(1, nh + 1)
        }
    )
    cat_to_idx = {c: i + 1 for i, c in enumerate(all_categories)}
    cat_idx = torch.zeros(nh, dtype=torch.long)
    for mi in range(1, nh + 1):
        cat = (
            pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("category", "")
            or "未知"
        )
        cat_idx[mi - 1] = cat_to_idx.get(cat, 0)

    # ── 药性索引 ──
    nat_idx = torch.zeros(nh, dtype=torch.long)
    for mi in range(1, nh + 1):
        nat = (
            pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("nature", "") or ""
        )
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
        flv = (
            pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]].get("flavor", "") or ""
        )
        found = False
        for fi, f_name in enumerate(FLAVOR_LIST, 1):
            if f_name in flv:
                flv_idx[mi - 1] = fi
                found = True
                break
        if not found:
            flv_idx[mi - 1] = len(FLAVOR_LIST) + 1  # unknown category

    # ── v2 多热特征：性/味/归经（一味药"甘、辛"两味同时表达；归经入模） ──
    nature_mh = torch.zeros(nh, len(NATURE_LIST))
    flavor_mh = torch.zeros(nh, len(FLAVOR_LIST))
    meridian_mh = torch.zeros(nh, len(MERIDIAN_LIST))
    for mi in range(1, nh + 1):
        herb = pipeline.herbs[pipeline.model_idx_to_herb_db_id[mi]]
        nat = herb.get("nature", "") or ""
        flv = herb.get("flavor", "") or ""
        mer = herb.get("meridian", "") or ""
        for ni, n_name in enumerate(NATURE_LIST):
            if n_name in nat:
                nature_mh[mi - 1, ni] = 1.0
        for fi, f_name in enumerate(FLAVOR_LIST):
            if f_name in flv:
                flavor_mh[mi - 1, fi] = 1.0
        for ci, m_name in enumerate(MERIDIAN_LIST):
            if m_name in mer:
                meridian_mh[mi - 1, ci] = 1.0

    logger.info(
        "药材特征: chars=%s, categories=%s (v2 多热: 性/味/归经)",
        num_chars, len(all_categories),
    )
    return {
        "char_multihot": char_mh,
        "category_idx": cat_idx,
        "nature_idx": nat_idx,
        "flavor_idx": flv_idx,
        "nature_multihot": nature_mh,
        "flavor_multihot": flavor_mh,
        "meridian_multihot": meridian_mh,
        "num_chars": num_chars,
        "num_categories": len(all_categories) + 1,  # +1 for padding
        "num_natures": len(NATURE_LIST) + 2,  # +1 padding +1 unknown
        "num_flavors": len(FLAVOR_LIST) + 2,
    }


# 训练器


class Trainer:
    def __init__(
        self,
        model: TCMNet,
        herb_features: dict[str, torch.Tensor],
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        grad_accum: int = 1,
        contrastive_weight: float = 0.2,
        label_smoothing: float = 0.05,
        toxic_herb_mask: Optional[torch.Tensor] = None,
        rank_loss_weight: float = 0.2,
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_accum = grad_accum
        self.contrastive_weight = contrastive_weight
        self.label_smoothing = label_smoothing
        self.rank_loss_weight = rank_loss_weight
        # 毒性药材掩码 [H]（bool）：训练期对毒性药负样本加权抑制，
        # 压低其被推荐的概率（与推理期安全硬拦截双保险）
        self.toxic_mask = (
            toxic_herb_mask.to(device) if toxic_herb_mask is not None else None
        )

        # 药材特征预加载到设备（v2：性味/归经 multi-hot）
        self.herb_char = herb_features["char_multihot"].to(device)
        self.herb_cat = herb_features["category_idx"].to(device)
        self.herb_nat = herb_features["nature_multihot"].to(device).float()
        self.herb_flv = herb_features["flavor_multihot"].to(device).float()
        self.herb_mer = herb_features["meridian_multihot"].to(device).float()

        # 损失函数
        self.syn_criterion = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing
        )
        self.formula_criterion = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing
        )

        # 优化器
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_params, lr=lr, betas=(0.9, 0.98)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
        )

        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        self.history = {"train": [], "val": [], "lr": []}
        self.global_step = 0

    def _herb_bce_loss(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """加权 BCE 损失（处理正负样本不平衡）

        标签约定：
          -  1.0 = 正样本（应推荐）
          - -1.0 = 负样本（不应推荐，参与训练，目标=0）
          -  0.0 = 未标注（不参与训练）
        """
        pos_mask = labels > 0.5  # 正样本
        neg_mask = labels < -0.5  # 负样本（先前被忽略，现已修复）
        ~pos_mask & ~neg_mask  # 0 = 未标注，忽略

        has_pos = pos_mask.any()
        has_neg = neg_mask.any()
        if not has_pos and not has_neg:
            return torch.tensor(0.0, device=self.device)

        # 参与 loss 计算的 mask
        active_mask = pos_mask | neg_mask
        # 目标值：正样本=1，负样本=0
        targets = torch.zeros_like(scores)
        targets[pos_mask] = 1.0

        # 权重：负样本权重 = 正样本数/负样本数（平衡），上限50
        if has_pos and has_neg:
            neg_weight_val = (
                pos_mask.float().sum() / neg_mask.float().sum().clamp(min=1)
            ).clamp(1, 50)
        else:
            neg_weight_val = 1.0

        weight = torch.ones_like(scores)
        weight[pos_mask] = 1.0
        weight[neg_mask] = weight.new_tensor(neg_weight_val)
        # 毒性药负样本额外抑制（权重 ≥20），从训练源头降低毒性药被推荐的可能
        if self.toxic_mask is not None:
            tm = self.toxic_mask.view(1, -1).expand_as(neg_mask)
            toxic_neg = neg_mask & tm
            if toxic_neg.any():
                weight[toxic_neg] = weight.new_tensor(max(float(neg_weight_val), 20.0))

        loss = F.binary_cross_entropy_with_logits(
            scores[active_mask],
            targets[active_mask],
            weight=weight[active_mask],
        )
        return loss

    def _rank_loss(self, scores: torch.Tensor, labels: torch.Tensor,
                   margin: float = 0.5) -> torch.Tensor:
        """pairwise 排序损失：正组均值 vs 最难负样本的 hinge。

        治"分数饱和"（实测 Top-12 全 1.000）：BCE 只压正类趋近 1，
        ranking 项额外要求正类分数高出最难负类一个 margin，
        让输出分数重新具备相对排序含义。
        """
        pos_mask = labels > 0.5
        neg_mask = labels < -0.5
        ok = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if ok.sum() == 0:
            return scores.new_zeros(())
        pos_mean = (scores * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        neg_hard = scores.masked_fill(~neg_mask, -1e4).max(dim=1).values
        loss = F.relu(margin - pos_mean + neg_hard)
        return loss[ok].mean()

    def train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()

        sym_ids = batch["symptom_ids"].to(self.device)
        sym_mask = batch["symptom_mask"].to(self.device)
        herb_lbl = batch["herb_labels"].to(self.device)
        form_lbl = batch["formula_label"].to(self.device)
        syn_lbl = batch["syndrome_label"].to(self.device)
        sample_weight = batch.get(
            "sample_weight", torch.ones(sym_ids.size(0), device=self.device)
        )
        # 样本权重归一到均值1，防止整体loss scale偏移，确保在同一设备
        sw = sample_weight.to(self.device) / sample_weight.to(self.device).mean().clamp(
            min=0.01
        )

        # 混合精度 (仅 CUDA)
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            syn_logits, herb_scores, form_scores, sv, hv = self.model(
                sym_ids,
                sym_mask,
                self.herb_char,
                self.herb_cat,
                self.herb_nat,
                self.herb_flv,
                self.herb_mer,
            )

            loss = 0.0
            metrics = {}

            # 证型分类
            mask_syn = syn_lbl != 0
            if mask_syn.sum() > 0:
                l_syn = self.syn_criterion(syn_logits, syn_lbl)
                if not torch.isnan(l_syn):
                    # 样本加权
                    l_syn = (
                        (l_syn * sw[mask_syn]).mean() if mask_syn.sum() > 1 else l_syn
                    )
                    loss += l_syn
                    metrics["syn_loss"] = l_syn.item()

            # 方剂排序
            mask_form = form_lbl > 0
            if mask_form.sum() > 0:
                l_form = self.formula_criterion(
                    form_scores[mask_form], form_lbl[mask_form]
                )
            else:
                l_form = self.formula_criterion(form_scores, form_lbl)
            if not torch.isnan(l_form):
                loss += l_form * 0.5
                metrics["form_loss"] = l_form.item()

            # 药材多标签（加权 BCE）
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

            # 排序损失（治分数饱和：正组均值须高出最难负样本 margin）
            l_rank = self._rank_loss(herb_scores, herb_lbl)
            if not torch.isnan(l_rank) and l_rank.item() > 0:
                loss += l_rank * self.rank_loss_weight
                metrics["rank_loss"] = l_rank.item()

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
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
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
                sym_ids,
                sym_mask,
                self.herb_char,
                self.herb_cat,
                self.herb_nat,
                self.herb_flv,
                self.herb_mer,
            )

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
                    _, topk = torch.topk(
                        herb_scores[i], k=min(10, herb_scores.shape[1])
                    )
                    herb_hits10 += sum(
                        1 for idx in topk.tolist() if pos_mask[idx].item()
                    )
                    herb_total_pos += pos_mask.sum().item()

            # 方剂
            mask_form = form_lbl > 0
            if mask_form.sum() > 0:
                l_f = self.formula_criterion(
                    form_scores[mask_form], form_lbl[mask_form]
                )
            else:
                l_f = self.formula_criterion(form_scores, form_lbl)
            if not torch.isnan(l_f):
                total_form_loss += l_f.item()

            n_batches += 1

        n = max(n_batches, 1)
        return {
            "val_loss": (total_syn_loss + total_herb_loss * 0.3 + total_form_loss * 0.5)
            / n,
            "syn_acc": syn_correct / max(syn_total, 1),
            "herb_recall10": herb_hits10 / max(herb_total_pos, 1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        save_dir: str = "tcm_recommender",
        save_prefix: str = "best_model",
    ):
        best_val_loss = float("inf")
        patience = 0
        early_stop_patience = 15
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_metrics = {
                "syn_loss": 0,
                "form_loss": 0,
                "herb_loss": 0,
                "cont_loss": 0,
                "total_loss": 0,
            }
            n_steps = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            last_batch_idx = None
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                n_steps += 1

                if (batch_idx + 1) % self.grad_accum == 0:
                    self.optimizer_step()

                pbar.set_postfix(
                    {
                        k: f"{v / n_steps:.3f}"
                        for k, v in epoch_metrics.items()
                        if k in ("total_loss", "syn_loss")
                    }
                )
                last_batch_idx = batch_idx

            # 处理最后的不完整梯度累积
            # 空训练集时 last_batch_idx 为 None，跳过以避免 NameError
            if last_batch_idx is not None and (last_batch_idx + 1) % self.grad_accum != 0:
                self.optimizer_step()

            # 验证
            val_metrics = self.evaluate(val_loader)
            epoch_time = time.time() - epoch_start

            logger.info(
                "Epoch %3d | tr_loss=%.4f | val=%.4f syn_acc=%.3f herb_r@10=%.3f | time=%.1fs lr=%.2e",
                epoch + 1,
                epoch_metrics["total_loss"] / n_steps,
                val_metrics["val_loss"],
                val_metrics["syn_acc"],
                val_metrics["herb_recall10"],
                epoch_time,
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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "history": self.history,
                "config": {
                    "d_model": self.model.d_model,
                    "num_layers": len(list(self.model.symptom_encoder.layers)),
                    "version": "v3-compositional",
                },
            },
            path,
        )
        logger.info("模型已保存: %s", path)


# 主函数


def main():
    parser = argparse.ArgumentParser(description="TCM 训练")
    parser.add_argument("--db", default="data/tcm_knowledge.db")
    parser.add_argument("--data", default="data/train_data.npz")
    parser.add_argument("--output", default="tcm_recommender/best_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--samples",
        type=int,
        default=500000,
        help="生成的训练样本数（如果 npz 不存在则生成）",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument(
        "--d-model",
        type=int,
        default=384,
        help="模型维度 (512 for 8M params, 384 for ~5M)",
    )
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.2,
        help="pairwise 排序损失权重（治分数饱和）",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("设备: %s", device)

    # 1. 数据
    data_path = Path(args.data)
    if not data_path.exists():
        logger.info("生成训练数据 (%s 样本)...", args.samples)
        pipeline = TCMDataPipeline(args.db)
        pipeline.load_all()
        data = pipeline.generate_samples(args.samples)
        pipeline.save(data, str(data_path))
        pipeline.close()
        data = dict(
            np.load(data_path, allow_pickle=True)
        )  # 重新加载（含 herb_features 缓存）
    else:
        logger.info("加载已有数据: %s", data_path)
        data = dict(np.load(data_path, allow_pickle=True))

    n_samples = int(data["num_samples"])
    n_symptoms = int(data["num_symptoms"])
    n_herbs = int(data["num_herbs"])
    n_formulas = int(data["num_formulas"])
    n_syndromes = int(data["num_syndromes"])
    logger.info(
        "数据: %s 样本, %s 症状, %s 药材, %s 方剂, %s 证型",
        n_samples,
        n_symptoms,
        n_herbs,
        n_formulas,
        n_syndromes,
    )

    # 检查 npz 字段完整性
    if "sample_weights" not in data:
        logger.warning(
            "npz 缺少 sample_weights 字段（旧数据格式），已自动回退为全1权重。"
            "建议用 data_pipeline 重新生成数据以获得样本加权功能。"
        )
    if "herb_features" not in data:
        logger.info(
            "npz 缺少 herb_features 缓存（旧数据格式），训练时将连接数据库构建特征。"
        )

    # 划分
    dataset = TCMDataset(data)
    n_train = int(n_samples * 0.85)
    n_val = n_samples - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info("训练集: %s | 验证集: %s", n_train, n_val)

    # 2. 特征（优先从 npz 加载缓存，无需重复连接数据库；缺 v2 multi-hot 时回退重建）
    herb_feat_data = data.get("herb_features")
    hf_has_v2 = (
        herb_feat_data is not None
        and "nature_multihot" in (herb_feat_data.item() or {})
    )
    if herb_feat_data is not None and hf_has_v2:
        logger.info("从 npz 加载缓存的药材特征...")
        _hf = herb_feat_data.item()
        herb_feat = {
            "char_multihot": _to_tensor(_hf["char_multihot"], torch.float32),
            "category_idx": _to_tensor(_hf["category_idx"], torch.long),
            "nature_multihot": _to_tensor(_hf["nature_multihot"], torch.float32),
            "flavor_multihot": _to_tensor(_hf["flavor_multihot"], torch.float32),
            "meridian_multihot": _to_tensor(_hf["meridian_multihot"], torch.float32),
            "num_chars": int(_hf["num_chars"]),
            "num_categories": int(_hf["num_categories"]),
            "num_natures": int(_hf["num_natures"]),
            "num_flavors": int(_hf["num_flavors"]),
        }
    else:
        if herb_feat_data is not None:
            logger.warning(
                "npz 药材特征缺 v2 multi-hot（归经/多味），改用当前数据库重建..."
            )
        else:
            logger.info(
                "构建药材特征（npz 中无缓存，建议用 data_pipeline 重新生成以加速训练）..."
            )
        pipeline = TCMDataPipeline(args.db)
        pipeline.load_all()
        herb_feat = build_herb_features(pipeline)
        pipeline.close()

    # 2.5 一致性守卫 + 组成矩阵 + 毒性掩码（组合式方剂表征，始终以当前库为准）
    logger.info("解析方剂组成（组合式方剂表征）...")
    pipe = TCMDataPipeline(args.db)
    pipe.load_all()
    if (pipe.num_symptoms, pipe.num_herbs, pipe.num_formulas, pipe.num_syndromes) != (
        n_symptoms, n_herbs, n_formulas, n_syndromes
    ):
        pipe.close()
        raise SystemExit(
            "训练数据 data/train_data.npz 与当前数据库实体数不一致，直接训练会得到与库错位的模型。\n"
            f"  npz(sym={n_symptoms},herb={n_herbs},form={n_formulas},syn={n_syndromes})\n"
            f"  db(sym={pipe.num_symptoms},herb={pipe.num_herbs},form={pipe.num_formulas},syn={pipe.num_syndromes})\n"
            "请删除 data/train_data.npz 后重新运行训练（将按当前库自动重新生成数据）。"
        )
    formula_comp = pipe.build_formula_composition()
    herb_names_for_mask = [
        pipe.herbs[pipe.model_idx_to_herb_db_id[i]]["name"]
        for i in range(1, pipe.num_herbs + 1)
    ]
    pipe.close()
    toxic_mask = torch.tensor(
        [is_toxic_herb(n) for n in herb_names_for_mask], dtype=torch.bool
    )
    logger.info("毒性药材掩码: %d/%d 味", int(toxic_mask.sum()), len(toxic_mask))

    # 3. 模型
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
        formula_composition=_to_tensor(formula_comp, torch.float32),
    )

    # 4. 训练
    trainer = Trainer(
        model,
        herb_feat,
        device,
        lr=args.lr,
        grad_accum=args.grad_accum,
        contrastive_weight=args.contrastive_weight,
        toxic_herb_mask=toxic_mask,
        rank_loss_weight=args.rank_loss_weight,
    )
    trainer.train(
        train_loader, val_loader, epochs=args.epochs, save_prefix="best_model"
    )

    # 5. 最终测试
    logger.info("最终测试...")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # 简单的推理测试
    val_metrics = trainer.evaluate(val_loader)
    logger.info(
        "最终验证: loss=%.4f syn_acc=%.3f herb_r@10=%.3f",
        val_metrics["val_loss"],
        val_metrics["syn_acc"],
        val_metrics["herb_recall10"],
    )

    # 保存最终结果
    results_path = Path(args.output).with_suffix(".json")
    results_path.write_text(
        json.dumps(
            {
                "val_metrics": {
                    k: float(v)
                    if isinstance(v, (np.floating, float, np.integer, int))
                    else v
                    for k, v in val_metrics.items()
                },
                "config": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    logger.info("全部完成!")


if __name__ == "__main__":
    main()
