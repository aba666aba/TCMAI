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
"""路径解析：只读资产根目录与可写数据目录分离。

发布包通常安装在 Program Files 等需要管理员权限的只读位置。
运行时产生的数据（历史记录、日志、知识库可变副本）必须落到
普通用户可写的目录，否则会静默写失败或触发 UAC 虚拟化，
且卸载时会随 {app} 目录一起被删除。

规则：
- 应用根目录（只读资产）：TCMAI_ROOT 环境变量 > frozen exe 所在目录 > 源码根目录
- 可写数据目录：
    1. TCMAI_DATA_DIR 环境变量（测试/自定义部署用）
    2. 打包环境（frozen）：Windows 下 %LOCALAPPDATA%/TCMAI/data，
       其他平台 ~/.local/share/TCMAI/data
    3. 开发环境：沿用项目内 data/（保持开发工作流与现有测试不变）

知识库采用"播种"策略：首次启动时把安装目录内的初始 tcm_knowledge.db
复制到可写数据目录，之后所有读写（含知识管理页的用户编辑）都针对副本。
旧版本写在安装目录 data/ 下的 history.json 会在首次启动时迁移。
"""
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DB_FILE_NAME = "tcm_knowledge.db"
HISTORY_FILE_NAME = "history.json"


def get_app_root() -> Path:
    """获取只读资产根目录（代码、模型、种子数据）。"""
    tcm_root = os.environ.get("TCMAI_ROOT")
    if tcm_root:
        return Path(tcm_root)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _default_writable_base() -> Path:
    """平台默认可写基目录（不含 data 子目录）。"""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "TCMAI"
        return Path.home() / "AppData" / "Local" / "TCMAI"
    # Linux/macOS 遵循 XDG 惯例
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "TCMAI"
    return Path.home() / ".local" / "share" / "TCMAI"


def get_data_dir() -> Path:
    """获取可写数据目录（历史记录、日志、知识库副本）。

    目录不存在时会尝试创建；创建失败（极端权限场景）则回退到
    应用根目录下的 data/，由调用方按原逻辑容忍不可写。
    """
    override = os.environ.get("TCMAI_DATA_DIR")
    if override:
        data_dir = Path(override)
    elif getattr(sys, "frozen", False):
        data_dir = _default_writable_base() / "data"
    else:
        # 开发环境：沿用项目内 data/，保持现有工作流与测试不变
        data_dir = get_app_root() / "data"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # 创建权限探测：不可写目录 mkdir 不报错，写入才报错
        probe = data_dir / ".write_probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return data_dir
    except OSError as exc:
        fallback = get_app_root() / "data"
        logger.warning("可写数据目录不可用(%s: %s)，回退到 %s", type(exc).__name__, exc, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def get_log_dir() -> Path:
    """获取日志目录（可写）。"""
    log_dir = get_data_dir() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback = get_app_root() / "data" / "logs"
        logger.warning("日志目录创建失败(%s)，回退到 %s", exc, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return log_dir


def get_history_file() -> Path:
    """获取历史记录文件路径（可写目录内）。"""
    return get_data_dir() / HISTORY_FILE_NAME


def ensure_database() -> Path | None:
    """确保可写知识库副本存在，返回其路径。

    - 副本已存在：直接返回
    - 副本不存在且安装目录内有种子库：复制（播种）后返回
    - 两处都没有：返回 None（KnowledgeBase 会降级为空知识库）

    同时把旧版本写在安装目录 data/ 下的 history.json 迁移到可写目录
    （仅当目标不存在时复制，不删除原文件）。
    """
    data_dir = get_data_dir()
    writable_db = data_dir / DB_FILE_NAME

    if not writable_db.exists():
        seed_db = get_app_root() / "data" / DB_FILE_NAME
        if seed_db.exists() and seed_db.resolve() != writable_db.resolve():
            try:
                shutil.copy2(seed_db, writable_db)
                logger.info("已播种知识库副本: %s -> %s", seed_db, writable_db)
            except OSError as exc:
                logger.error("知识库播种失败: %s", exc)
                return seed_db if seed_db.exists() else None

    if not writable_db.exists():
        return None

    _migrate_legacy_files(data_dir)
    return writable_db


def _migrate_legacy_files(data_dir: Path) -> None:
    """把旧版本写在安装目录 data/ 下的用户数据迁移到可写目录。"""
    legacy_dir = get_app_root() / "data"
    if legacy_dir.resolve() == data_dir.resolve():
        return  # 开发环境两处相同，无需迁移
    legacy_history = legacy_dir / HISTORY_FILE_NAME
    target_history = data_dir / HISTORY_FILE_NAME
    if legacy_history.exists() and not target_history.exists():
        try:
            shutil.copy2(legacy_history, target_history)
            logger.info("已迁移历史记录: %s -> %s", legacy_history, target_history)
        except OSError as exc:
            logger.warning("历史记录迁移失败（不影响启动）: %s", exc)
