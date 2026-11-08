# -*- coding: utf-8 -*-
"""零依赖中文分词器。

默认使用内置的 CJK 字/二元 + 拉丁词切分（无需任何第三方库）。
若环境中可用 jieba，可通过 use_jieba=True 启用以获得更准的切分。
"""
import re

_CJK = re.compile(r"[一-鿿㐀-䶿]")
_ASCII = re.compile(r"[a-zA-Z0-9]+")

try:  # jieba 为可选增强，缺失不影响基本功能
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def tokenize(text, use_jieba=None):
    """将文本切分为检索词列表。

    - 拉丁字母/数字连续串作为独立 token；
    - 中文按单字 + 相邻二元切分（兼顾召回与精度）；
    - 可选 jieba 用于中文（仍保留拉丁 token）。
    """
    if not text:
        return []
    text = str(text).lower()

    # 拉丁词
    ascii_tokens = _ASCII.findall(text)

    # 中文内容
    cjk = "".join(_CJK.findall(text))

    if use_jieba is None:
        use_jieba = _HAS_JIEBA

    if use_jieba and _HAS_JIEBA:
        cjk_tokens = [t for t in jieba.cut(cjk) if t.strip()]
        # 过滤掉纯标点/单空格
        cjk_tokens = [t for t in cjk_tokens if _CJK.search(t) or _ASCII.search(t)]
    else:
        cjk_tokens = []
        for i, ch in enumerate(cjk):
            cjk_tokens.append(ch)
            if i + 1 < len(cjk):
                cjk_tokens.append(cjk[i : i + 2])

    return ascii_tokens + cjk_tokens


def expand_query(text, synonym_maps=None):
    """在原始切词基础上，利用同义词/别名映射做查询扩展，提升召回。

    synonym_maps: 可传入 {词: [同义词...]} 结构的字典；缺省时返回原词。
    """
    base = tokenize(text)
    if not synonym_maps:
        return base
    expanded = list(base)
    for tok in base:
        for key, syns in synonym_maps.items():
            if tok == key or tok in syns:
                expanded.extend(syns)
                if key != tok:
                    expanded.append(key)
    # 去重保序
    seen = set()
    out = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
