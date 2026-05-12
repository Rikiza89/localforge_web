"""
BM25 アダプター — rank-bm25 を使ったコード特化型キーワード検索。
ChromaDB / sentence-transformers が使用不可な場合の高速フォールバックとして機能する。
ファイルパス・サマリー・コンテンツ先頭 300 文字を対象にインデックスを構築する。
"""

from __future__ import annotations

import logging
import re
from typing import List

from localforge.domain.models import FileChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """
    コード向けトークナイザー。
    - パス区切り文字・記号で分割
    - キャメルケースをスペースで分割 (fooBar → foo bar)
    - 短すぎるトークン（1文字）は除外
    """
    # パス区切り・記号で分割
    parts = re.split(r'[\s/\\_.\-:,;()\[\]{}\'\"<>|]+', text.lower())
    expanded: List[str] = []
    for part in parts:
        # キャメルケース分割: fooBar → foo Bar, FOOBar → FOO Bar
        split = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
        expanded.extend(split.split())
    return [t for t in expanded if len(t) > 1]


def get_top_chunks_bm25(
    chunks: List[FileChunk],
    query: str,
    top_n: int = 10,
) -> List[FileChunk]:
    """
    BM25Okapi スコアで上位 N 件の FileChunk を返す。
    rank-bm25 がインストールされていない場合は改良キーワードカウントにフォールバックする。

    コーパスはインデックス構築時に毎回ビルドする（メモリ上の chunks から数 ms で完了）。

    Args:
        chunks: 全 FileChunk リスト
        query: 検索クエリ文字列
        top_n: 返す件数

    Returns:
        BM25 スコア順の上位 FileChunk リスト
    """
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_n]

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.debug("rank-bm25 未インストール: キーワードカウントフォールバックを使用")
        return _keyword_fallback(chunks, query_tokens, top_n)

    # コーパス: パス + サマリー + コンテンツ先頭 300 文字をトークナイズ
    corpus = [
        _tokenize(f"{c.path} {c.summary or ''} {c.content[:300]}")
        for c in chunks
    ]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    # スコア降順で上位 N 件を返す（全件スコア 0 でも先頭 N 件を返す）
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in indexed[:top_n]]


def _keyword_fallback(
    chunks: List[FileChunk],
    query_tokens: List[str],
    top_n: int,
) -> List[FileChunk]:
    """rank-bm25 未インストール時の改良キーワードカウントフォールバック。"""
    query_set = set(query_tokens)

    def score(c: FileChunk) -> int:
        tokens = set(_tokenize(f"{c.path} {c.summary or ''} {c.content[:300]}"))
        return len(query_set & tokens)

    return sorted(chunks, key=score, reverse=True)[:top_n]
