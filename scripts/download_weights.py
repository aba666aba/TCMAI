#!/usr/bin/env python3
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

"""下载运行 TCMAI 所需的模型文件（这些文件体积大，不入 git 仓库）。

仓库里缺少的两类文件，本脚本负责补齐：

  1. ``tcm_recommender/best_model.pth``
     本项目自行训练的推荐模型权重（TCMNet，11.08M 参数，~127 MB）。
     它不在 git 仓库中（被 .gitignore 排除），托管在 ModelScope 模型库。

  2. ``models/nlu/qwen/``
     NLU 引擎使用的底座大模型 Qwen2.5-0.5B-Instruct（~942 MB）。
     底座模型不是本项目产物，直接从 ModelScope 公开模型库拉取即可。

用法::

    python scripts/download_weights.py              # 下载权重 + 底座模型
    python scripts/download_weights.py --only ckpt  # 只下载训练权重
    python scripts/download_weights.py --only base  # 只下载底座模型
    python scripts/download_weights.py --force      # 忽略已存在文件，强制重下
    python scripts/download_weights.py --verify-only  # 只校验本地文件，不联网

依赖：``modelscope``（``pip install modelscope``）。

说明：底座模型下载器会把 ``*.safetensors`` 之外的元数据（config.json /
tokenizer.json / chat_template.jinja 等）一并放在 ``models/nlu/qwen/``，
与 ``src/nlu/engine.py`` 的默认加载路径一致，不需要手工搬文件。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

try:  # 保证在仓库任意位置执行时都能定位到项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # pragma: no cover
    PROJECT_ROOT = Path.cwd()

# ── 可配置项（可用环境变量覆盖，方便权重换版本后无需改代码）──
CKPT_REPO = os.environ.get("TCMAI_CKPT_REPO", "aba666aba/TCMAI-tcm-recommender")
CKPT_FILE = os.environ.get("TCMAI_CKPT_FILE", "best_model.pth")
CKPT_SHA256 = os.environ.get(
    "TCMAI_CKPT_SHA256",
    "1d8414c676144652e1d3ce21f18ede039f56e01c135428a545739b3ca17da98f",
)
# 权重至少应有这么大，小于此值说明是 LFS 指针或下载被截断
CKPT_MIN_BYTES = 100 * 1024 * 1024

# 备用直链：当 ModelScope 不可用（未发布/网络不通/需要登录）时回退到该 URL。
# 典型取值是 GitHub Release 附件地址，例如
#   https://github.com/<owner>/<repo>/releases/download/<tag>/best_model.pth
CKPT_URL_FALLBACK = os.environ.get("TCMAI_CKPT_URL", "").strip()

BASE_REPO = os.environ.get("TCMAI_BASE_REPO", "Qwen/Qwen2.5-0.5B-Instruct")
BASE_MIN_SAFETENSORS_BYTES = 500 * 1024 * 1024

CKPT_DEST = PROJECT_ROOT / "tcm_recommender" / CKPT_FILE
BASE_DEST = PROJECT_ROOT / "models" / "nlu" / "qwen"

# 只保留底座模型的推理必需文件，避免把 .gitattributes 等仓库杂项搬进项目目录。
# 实测 Qwen/Qwen2.5-0.5B-Instruct 的文件清单为：
#   config.json, configuration.json, generation_config.json, tokenizer_config.json,
#   tokenizer.json, vocab.json, merges.txt, LICENSE, README.md, model.safetensors
# 注意：该模型库**不含** chat_template.jinja（对话模板内嵌在 tokenizer_config.json 中），
# 但完整权重目录（从本仓库 .msc 记录看）曾经包含它，故下载后仅在存在时提示。
BASE_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "LICENSE",
    "README.md",
]

# 判定底座模型是否可用的必需文件
BASE_REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def require_modelscope():
    try:
        from modelscope import snapshot_download  # noqa: F401
    except ImportError:
        print("[X] 缺少依赖 modelscope。请先安装：pip install modelscope")
        sys.exit(2)
    from modelscope import snapshot_download

    return snapshot_download


def download_url(url: str, dest: Path) -> bool:
    """从直链流式下载到 dest（用于 GitHub Release 等备用通道）。"""
    import urllib.error
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TCMAI-downloader"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    pct = done * 100 // total
                    print(f"\r    下载中 {pct:3d}%  {human(done)} / {human(total)}", end="")
                else:
                    print(f"\r    下载中 {human(done)}", end="")
        print()
    except (urllib.error.URLError, OSError) as exc:
        print(f"\n    [X] 直链下载失败：{type(exc).__name__}: {exc}")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    return True


def download_ckpt(force: bool) -> bool:
    print(f"\n=== [1/2] 训练权重 {CKPT_FILE} ===")
    print(f"    来源 : ModelScope 模型库 {CKPT_REPO}")
    if CKPT_URL_FALLBACK:
        print(f"    备用 : {CKPT_URL_FALLBACK}")

    if CKPT_DEST.exists() and not force:
        size = CKPT_DEST.stat().st_size
        if size >= CKPT_MIN_BYTES:
            print(f"    [跳过] 已存在 {CKPT_DEST} ({human(size)})")
            return True
        print(f"    [!] 已存在但体积异常偏小 ({human(size)})，重新下载")

    CKPT_DEST.parent.mkdir(parents=True, exist_ok=True)

    # 通道 1：ModelScope 模型库
    try:
        snapshot_download = require_modelscope()
    except SystemExit:
        snapshot_download = None

    if snapshot_download is not None:
        try:
            snapshot_download(
                CKPT_REPO,
                allow_file_pattern=[CKPT_FILE],
                local_dir=str(CKPT_DEST.parent),
            )
            if CKPT_DEST.exists() and CKPT_DEST.stat().st_size >= CKPT_MIN_BYTES:
                size = CKPT_DEST.stat().st_size
                print(f"    [OK] {CKPT_DEST} ({human(size)})")
                return True
            print("    [!] ModelScope 通道未得到有效权重文件")
        except Exception as exc:  # 网络/权限/模型库不存在
            print(f"    [!] ModelScope 通道失败：{type(exc).__name__}: {str(exc)[:160]}")
            print("        - 若提示模型库不存在，请确认权重已发布，或设置 TCMAI_CKPT_REPO")
            print("        - 私有模型库需先执行：modelscope login --token <你的Token>")

    # 通道 2：备用直链（GitHub Release 等）
    if CKPT_URL_FALLBACK:
        print(f"    → 回退到备用直链…")
        if download_url(CKPT_URL_FALLBACK, CKPT_DEST):
            size = CKPT_DEST.stat().st_size
            print(f"    [OK] {CKPT_DEST} ({human(size)})")
            return True
        return False

    print("    [X] 无可用下载通道。可设置环境变量 TCMAI_CKPT_URL 指向备用直链后重试。")
    return False


def verify_ckpt(strict: bool = True) -> bool:
    print(f"\n=== 校验训练权重 {CKPT_FILE} ===")
    if not CKPT_DEST.exists():
        print(f"    [X] 文件不存在：{CKPT_DEST}")
        return False

    size = CKPT_DEST.stat().st_size
    if size < CKPT_MIN_BYTES:
        print(f"    [X] 体积异常（{human(size)}），疑似 LFS 指针或下载未完成")
        return False
    print(f"    [OK] 体积 {human(size)}")

    if not CKPT_SHA256:
        print("    [--] 未配置预期 SHA256，跳过哈希校验")
        return True

    actual = sha256_of(CKPT_DEST)
    if actual.lower() != CKPT_SHA256.lower():
        print(f"    [X] SHA256 不匹配\n        期望 {CKPT_SHA256}\n        实际 {actual}")
        return not strict
    print(f"    [OK] SHA256 {actual}")
    return True


def verify_base() -> bool:
    print(f"\n=== 校验底座模型 {BASE_REPO} ===")
    if not BASE_DEST.exists():
        print(f"    [X] 目录不存在：{BASE_DEST}")
        return False

    weights = list(BASE_DEST.glob("*.safetensors"))
    if not weights:
        print(f"    [X] {BASE_DEST} 下没有 .safetensors 权重文件")
        return False

    ok = True
    for path in weights:
        size = path.stat().st_size
        flag = "[OK]" if size >= BASE_MIN_SAFETENSORS_BYTES else "[X]"
        if size < BASE_MIN_SAFETENSORS_BYTES:
            ok = False
        print(f"    {flag} {path.name} ({human(size)})")

    for name in BASE_REQUIRED_FILES:
        exists = (BASE_DEST / name).exists()
        ok = ok and exists
        print(f"    {'[OK]' if exists else '[X]'} {name}")

    # chat_template.jinja 在公开模型库中不存在（模板内嵌于 tokenizer_config.json），
    # 因此只作提示，不影响校验结论
    template = BASE_DEST / "chat_template.jinja"
    print(f"    [{'OK' if template.exists() else '--'}] chat_template.jinja（可选）")
    return ok


def download_base(force: bool) -> bool:
    print(f"\n=== [2/2] 底座模型 {BASE_REPO} ===")
    print(f"    目标目录 : {BASE_DEST}")
    print("    说明     : 底座模型不是本项目训练产物，公开模型库直接拉取")

    if BASE_DEST.exists() and not force:
        weights = [p for p in BASE_DEST.glob("*.safetensors") if p.stat().st_size >= BASE_MIN_SAFETENSORS_BYTES]
        if weights:
            print(f"    [跳过] 已存在权重 {weights[0].name} ({human(weights[0].stat().st_size)})")
            return True
        print("    [!] 目录存在但缺少完整权重，重新下载")

    BASE_DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download = require_modelscope()

    try:
        snapshot_download(
            BASE_REPO,
            allow_file_pattern=BASE_PATTERNS,
            local_dir=str(BASE_DEST),
        )
    except Exception as exc:
        print(f"    [X] 下载失败：{type(exc).__name__}: {exc}")
        print("        请检查网络，或改用 ModelScope 网页手工下载后解压到该目录")
        return False

    print(f"    [OK] 已下载到 {BASE_DEST}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载 TCMAI 运行所需的模型文件（训练权重 + 底座模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/download_weights.py\n"
            "  python scripts/download_weights.py --only ckpt --force\n"
            "  python scripts/download_weights.py --verify-only\n"
        ),
    )
    parser.add_argument(
        "--only",
        choices=["all", "ckpt", "base"],
        default="all",
        help="只下载指定部分（默认 all）",
    )
    parser.add_argument("--force", action="store_true", help="忽略已存在的文件，强制重新下载")
    parser.add_argument(
        "--verify-only", action="store_true", help="只校验本地文件，不联网下载"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="下载后不做 SHA256 校验（大文件校验需要额外读一遍磁盘，较慢）",
    )
    args = parser.parse_args()

    print("TCMAI 模型文件下载器")
    print(f"项目根目录 : {PROJECT_ROOT}")

    if args.verify_only:
        ok = True
        if args.only in ("all", "ckpt"):
            ok = verify_ckpt() and ok
        if args.only in ("all", "base"):
            ok = verify_base() and ok
        print("\n结论：", "全部校验通过" if ok else "存在缺失或异常，见上")
        return 0 if ok else 1

    ok = True
    if args.only in ("all", "ckpt"):
        ok = download_ckpt(args.force) and ok
        if ok and not args.no_verify:
            ok = verify_ckpt() and ok
    if args.only in ("all", "base"):
        ok = download_base(args.force) and ok

    print("\n" + "=" * 56)
    if ok:
        print("完成。现在可以启动系统：python 启动中医药智能问答系统.pyw")
    else:
        print("存在失败项，请按上面提示处理后重试。")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
