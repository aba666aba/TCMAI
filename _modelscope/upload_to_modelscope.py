"""发布 TCMAI 推荐权重到 ModelScope 模型库。

用法::

    python _modelscope/upload_to_modelscope.py --namespace <用户名>
    python _modelscope/upload_to_modelscope.py --namespace <用户名> --private
    python _modelscope/upload_to_modelscope.py --namespace <用户名> --dry-run

Token 来源（按优先级）：
  1. 环境变量 MODELSCOPE_API_TOKEN
  2. 本机 ~/.modelscope/credentials/session（modelscope login 写入）

安全约定：Token 只在进程内存中使用，绝不打印、绝不写入本仓库任何文件。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_CARD = PROJECT_ROOT / "_modelscope" / "MODEL_CARD.md"
SRC_CKPT = PROJECT_ROOT / "tcm_recommender" / "best_model.pth"
SRC_LICENSE = PROJECT_ROOT / "LICENSE"
CRED_FILE = Path(os.path.expanduser("~")) / ".modelscope" / "credentials" / "session"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_token() -> str:
    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if token:
        log("Token 来源：环境变量 MODELSCOPE_API_TOKEN")
        return token
    if CRED_FILE.exists():
        token = CRED_FILE.read_text(encoding="utf-8", errors="ignore").strip()
        if token:
            log(f"Token 来源：{CRED_FILE}（长度 {len(token)}，内容不显示）")
            return token
    log("[X] 未找到 AccessToken。")
    log("    请先执行：modelscope login --token <你的AccessToken>")
    log("    或设置环境变量 MODELSCOPE_API_TOKEN")
    sys.exit(2)


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="发布 TCMAI 权重到 ModelScope")
    parser.add_argument("--namespace", required=True, help="ModelScope 用户名（命名空间）")
    parser.add_argument("--repo-name", default="TCMAI-tcm-recommender", help="模型库名")
    parser.add_argument("--private", action="store_true", help="创建为私有模型库（默认公开）")
    parser.add_argument("--dry-run", action="store_true", help="只检查本地待传文件，不上传")
    args = parser.parse_args()

    model_id = f"{args.namespace}/{args.repo_name}"

    # ── 本地文件检查 ──
    missing = [p for p in (SRC_CARD, SRC_CKPT) if not p.exists()]
    if missing:
        for p in missing:
            log(f"[X] 缺少文件：{p}")
        return 2

    size = SRC_CKPT.stat().st_size
    digest = sha256_of(SRC_CKPT)
    log("待上传文件：")
    log(f"  best_model.pth  {size / 1024 / 1024:.1f} MB")
    log(f"  SHA256          {digest}")
    log(f"  MODEL_CARD.md   {SRC_CARD.stat().st_size} bytes")
    log(f"  LICENSE         {'有' if SRC_LICENSE.exists() else '无'}")
    log(f"目标模型库：{model_id}（{'私有' if args.private else '公开'}）")

    if args.dry_run:
        log("\n--dry-run：仅检查，未上传。")
        log(f"确认无误后请执行（去掉 --dry-run），上传后请在仓库中核对 SHA256 = {digest}")
        return 0

    token = load_token()
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        log("[X] 缺少依赖 modelscope：pip install modelscope")
        return 2

    api = HubApi()
    api.login(token)

    # ── 创建模型库（已存在则跳过）──
    # 先探测是否已存在，避免依赖服务端错误码（重名时返回
    # Code=10010202004「该名称已被注册使用」，容易被误判为失败）
    exists = False
    try:
        api.model_info(model_id)
        exists = True
        log(f"[--] 模型库 {model_id} 已存在，直接上传")
    except Exception:
        pass

    if not exists:
        try:
            api.create_model(
                model_id,
                visibility=1 if args.private else 5,
                token=token,
            )
            log(f"[OK] 已创建模型库 {model_id}")
        except Exception as exc:
            text = str(exc)
            if any(k in text for k in ("already", "existed", "10010202004", "100201010")):
                log(f"[--] 模型库 {model_id} 已存在，直接上传")
            else:
                log(f"[X] 创建模型库失败：{type(exc).__name__}: {text[:200]}")
                return 1

    # ── 组装待上传目录：ModelScope 以仓库根 README.md 作为模型卡 ──
    with tempfile.TemporaryDirectory(prefix="tcmai_ms_") as tmp:
        stage = Path(tmp)
        shutil.copy2(SRC_CARD, stage / "README.md")
        shutil.copy2(SRC_CKPT, stage / "best_model.pth")
        if SRC_LICENSE.exists():
            shutil.copy2(SRC_LICENSE, stage / "LICENSE")

        log("上传中（127 MB，请耐心等待）…")
        # 用 upload_folder（SDK 官方推荐，支持断点续传/并行/重试）。
        # 不要用 push_model：它已被标记 deprecated，且走 git 流程，在已存在
        # master 分支的模型库上必然失败（fatal: a branch named 'master' already exists）
        try:
            api.upload_folder(
                repo_id=model_id,
                folder_path=str(stage),
                commit_message="发布 TCMAI 推荐模型权重（TCMNet 11.08M 参数，syn_acc 0.979）",
                token=token,
                repo_type="model",
                revision="master",
            )
        except Exception as exc:
            log(f"[X] 上传失败：{type(exc).__name__}: {str(exc)[:300]}")
            return 1

    log(f"[OK] 上传完成：https://www.modelscope.cn/models/{model_id}")
    log(f"     权重直链：https://www.modelscope.cn/models/{model_id}/resolve/master/best_model.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
