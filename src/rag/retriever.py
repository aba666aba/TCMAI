# -*- coding: utf-8 -*-
"""混合检索器（纯 Python BM25 + 同义词扩展，零依赖）。

- 加载 corpus.py 产出的 data/rag_corpus.jsonl
- 默认 BM25（Okapi）检索；可选载入项目内置同义词/别名映射做查询扩展，提升召回
- 返回 top-K 证据文档，附带出处（type / title / meta），供答案合成器消费
"""
import json
import os
from collections import defaultdict

from . import tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_PATH = os.path.join(ROOT, "data", "rag_corpus.jsonl")

_K1 = 1.5
_B = 0.75


def _load_synonym_maps():
    """尽力加载项目内置同义词/别名映射用于查询扩展。"""
    try:
        import sys

        sys.path.insert(0, os.path.join(ROOT, "src"))
        from core.tcm_config import TCM_SYNONYM_MAP, SYMPTOM_ALIAS_MAP  # type: ignore

        merged = {}
        for k, v in (TCM_SYNONYM_MAP or {}).items():
            merged[k] = list(v)
        for k, v in (SYMPTOM_ALIAS_MAP or {}).items():
            merged.setdefault(k, [])
            if isinstance(v, (list, tuple, set)):
                merged[k].extend(v)
            else:
                merged[k].append(v)
        return merged
    except Exception:
        return {}


class Retriever:
    def __init__(self, corpus_path=CORPUS_PATH, use_synonyms=True):
        self.corpus_path = corpus_path
        self.docs = []
        self.doc_tokens = []
        self.N = 0
        self.avgdl = 0.0
        self.df = defaultdict(int)
        self.idf = {}
        self.synonym_maps = _load_synonym_maps() if use_synonyms else {}
        self.title_index = {}  # 归一化标题/别名 -> [doc_idx]，用于实体名精确置顶
        self._loaded = False

    @staticmethod
    def _norm_title(s):
        return (s or "").replace(" ", "").replace("\u3000", "")

    def load(self):
        if self._loaded:
            return self
        with open(self.corpus_path, encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f if line.strip()]

        tokenized = []
        for d in self.docs:
            toks = tokenizer.tokenize(d["text"])
            tokenized.append(toks)

        # 实体标题索引（标题 + 别名均入索引）
        tidx = defaultdict(list)
        for i, d in enumerate(self.docs):
            tidx[self._norm_title(d.get("title", ""))].append(i)
            meta = d.get("meta") or {}
            for a in (meta.get("aliases") or []) if isinstance(meta, dict) else []:
                tidx[self._norm_title(a)].append(i)
        self.title_index = dict(tidx)

        self.N = len(self.docs)
        lengths = [len(t) for t in tokenized]
        self.avgdl = (sum(lengths) / self.N) if self.N else 0.0

        df = defaultdict(int)
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        self.df = df
        self.idf = {
            t: __import__("math").log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }
        self.doc_tokens = tokenized
        self._loaded = True
        return self

    def _score_one(self, q_tokens, doc_toks):
        # 词频
        tf = defaultdict(int)
        for t in doc_toks:
            tf[t] += 1
        dl = len(doc_toks)
        score = 0.0
        for qt in set(q_tokens):
            if qt not in self.idf:
                continue
            f = tf.get(qt, 0)
            if f == 0:
                continue
            denom = f + _K1 * (1 - _B + _B * (dl / self.avgdl if self.avgdl else 1))
            score += self.idf[qt] * (f * (_K1 + 1)) / denom
        return score

    def search(self, query, top_k=8, expand=True):
        if not self._loaded:
            self.load()
        if expand and self.synonym_maps:
            q_tokens = tokenizer.expand_query(query, self.synonym_maps)
        else:
            q_tokens = tokenizer.tokenize(query)

        scored = []
        for i, doc_toks in enumerate(self.doc_tokens):
            s = self._score_one(q_tokens, doc_toks)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])

        # 实体名精确命中置顶：查询（去空格归一）与实体标题/别名完全一致时，
        # 该实体文档提到结果最前，避免被 token 密度更高的短文档（单味药）挤掉
        qn = self._norm_title(query)
        exact_ids = self.title_index.get(qn) or []
        if exact_ids:
            exact_set = set(exact_ids)
            head = [(s, i) for (s, i) in scored if i in exact_set]
            in_scored = {i for _, i in scored}
            for idx in exact_ids:
                if idx not in in_scored:
                    head.append((0.0, idx))
            tail = [(s, i) for (s, i) in scored if i not in exact_set]
            scored = head + tail

        # 类型多样性兜底：短症状查询易被证候/单味药"短文档"占满（BM25 长度偏置）。
        # 对 patent 与 formula 分别保证在结果中至少出现 1 个（若存在得分>0 者）；
        # 缺席时从结果尾部淘汰非实体文档腾位，不打乱头部顺序。
        top_slice = scored[:top_k]
        rest = scored[top_k:]

        def _dtype(i):
            return self.docs[i]["type"]

        fills = []
        in_slice_ids = {i for _, i in top_slice}
        for t in ("patent", "formula"):
            if any(_dtype(i) == t for _, i in top_slice) or any(
                _dtype(i) == t for _, i in fills
            ):
                continue
            for s, i in rest:
                if _dtype(i) == t and i not in in_slice_ids:
                    fills.append((s, i))
                    break
        if fills:
            kept = list(top_slice)
            for _ in fills:
                for k in range(len(kept) - 1, -1, -1):
                    if _dtype(kept[k][1]) not in ("patent", "formula"):
                        del kept[k]
                        break
            top_slice = kept + fills

        results = []
        for s, i in top_slice:
            d = self.docs[i]
            results.append(
                {
                    "score": round(s, 4),
                    "id": d["id"],
                    "type": d["type"],
                    "title": d["title"],
                    "text": d["text"],
                    "meta": d.get("meta", {}),
                }
            )
        return results


def main():
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "失眠 乏力 头晕"
    r = Retriever().load()
    print(f"查询: {q}\n")
    for hit in r.search(q, top_k=8):
        print(f"[{hit['type']:<8}] {hit['title']}  (score={hit['score']})")
        print(f"    {hit['text'][:80]}...")


if __name__ == "__main__":
    main()
