"""
Data preparation for HotpotQA (distractor setting).

Each HotpotQA example carries the question, the gold answer, and a small set
of paragraphs (typically 10: 2 gold + 8 distractors), each tagged with a
Wikipedia article title. We use the *distractor* setting because:

1. It needs no external Wikipedia retriever.
2. The paragraph titles are natural entity names, which is exactly what the
   graph_db memory mode needs as `entity` fields.

Loading paths:
- `prepare_hotpotqa_data(json_path=...)`: load from a local JSON file
  (the canonical HotpotQA distractor file: hotpot_dev_distractor_v1.json)
- `prepare_hotpotqa_data()`: looks for $HOTPOTQA_DATA env var pointing to
  a directory containing hotpot_train_v1.1.json and
  hotpot_dev_distractor_v1.json.

Both paths register the result with DatasetRegistry under name="hotpotqa".
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

from src.data.dataset import DatasetRegistry

logger = logging.getLogger(__name__)


_TRAIN_FILE_NAMES = ("hotpot_train_v1.1.json",)
_DEV_FILE_NAMES = ("hotpot_dev_distractor_v1.json",)


def get_hotpotqa_data_path() -> str:
    """Read $HOTPOTQA_DATA. Raise a clear error if unset/missing."""
    p = os.environ.get("HOTPOTQA_DATA")
    if not p:
        raise EnvironmentError(
            "HOTPOTQA_DATA environment variable not set. "
            "Set it to a directory containing hotpot_dev_distractor_v1.json "
            "(and optionally hotpot_train_v1.1.json)."
        )
    if not os.path.isdir(p):
        raise FileNotFoundError(f"HOTPOTQA_DATA directory not found: {p}")
    return p


def _normalize_example(raw: dict) -> dict:
    """Convert one raw HotpotQA example to MemexRL task dict.

    Raw HotpotQA shape:
        {
          "_id": str,
          "question": str,
          "answer": str,
          "type": str,                  # bridge / comparison
          "level": str,                 # easy / medium / hard
          "supporting_facts": [[title, sentence_idx], ...],
          "context": [[title, [sentence, sentence, ...]], ...],   # ~10 paragraphs
        }

    Output task dict:
        {
          "task_id": str,
          "question": str,
          "answer": str,
          "type": "bridge" | "comparison",
          "level": "easy"|"medium"|"hard",
          "passages": [{"title": str, "sentences": [str], "is_gold": bool}, ...],
          "supporting_facts": [{"title": str, "sentence_idx": int}, ...],
          "data_source": "hotpotqa",
          "max_steps": 30,
        }
    """
    sup_titles: set[str] = set(t for t, _ in raw.get("supporting_facts", []))
    passages = []
    for entry in raw.get("context", []):
        if not (isinstance(entry, list) and len(entry) == 2):
            continue
        title, sentences = entry
        if not isinstance(title, str) or not isinstance(sentences, list):
            continue
        passages.append({
            "title": title,
            "sentences": [str(s) for s in sentences],
            "is_gold": title in sup_titles,
        })

    sup_facts = [
        {"title": t, "sentence_idx": int(i)}
        for t, i in raw.get("supporting_facts", [])
        if isinstance(t, str)
    ]

    return {
        "task_id": str(raw.get("_id", "")),
        "question": str(raw.get("question", "")),
        "answer": str(raw.get("answer", "")),
        "type": str(raw.get("type", "")),
        "level": str(raw.get("level", "")),
        "passages": passages,
        "supporting_facts": sup_facts,
        "data_source": "hotpotqa",
        "max_steps": 30,
    }


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_first(dir_path: Path, candidates: tuple[str, ...]) -> Optional[Path]:
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    return None


def prepare_hotpotqa_data(
    data_dir: Optional[str] = None,
    json_path: Optional[str] = None,
    max_train_size: Optional[int] = None,
    max_test_size: Optional[int] = None,
    seed: int = 42,
) -> tuple:
    """Load HotpotQA distractor data and register with DatasetRegistry.

    Provide EITHER:
      - `json_path` to a single .json file (registered as the 'test' split)
      - `data_dir` containing the canonical filenames (registers train+test
        splits when both are present)
      - neither, in which case $HOTPOTQA_DATA is consulted as the data_dir.
    """
    train_tasks: list[dict] = []
    test_tasks: list[dict] = []

    if json_path is not None:
        # Single file → register as 'test' split (most common eval workflow)
        raw = _load_json(Path(json_path))
        test_tasks = [_normalize_example(r) for r in raw]
        logger.info(f"Loaded {len(test_tasks)} examples from {json_path}")
    else:
        if data_dir is None:
            data_dir = get_hotpotqa_data_path()
        d = Path(data_dir)

        train_p = _resolve_first(d, _TRAIN_FILE_NAMES)
        test_p = _resolve_first(d, _DEV_FILE_NAMES)
        if test_p is None:
            raise FileNotFoundError(
                f"No HotpotQA dev file in {d}. Expected one of: {_DEV_FILE_NAMES}"
            )
        if train_p is not None:
            train_tasks = [_normalize_example(r) for r in _load_json(train_p)]
            logger.info(f"Loaded {len(train_tasks)} train examples from {train_p}")
        test_tasks = [_normalize_example(r) for r in _load_json(test_p)]
        logger.info(f"Loaded {len(test_tasks)} test examples from {test_p}")

    rng = random.Random(seed)
    rng.shuffle(train_tasks)
    rng.shuffle(test_tasks)

    if max_train_size is not None:
        train_tasks = train_tasks[:max_train_size]
    if max_test_size is not None:
        test_tasks = test_tasks[:max_test_size]

    train_dataset = (
        DatasetRegistry.register_dataset("hotpotqa", train_tasks, "train")
        if train_tasks else None
    )
    test_dataset = DatasetRegistry.register_dataset("hotpotqa", test_tasks, "test")
    return train_dataset, test_dataset


# Standard HotpotQA EM/F1 helpers (mirrors official eval script)

import re
import string
from collections import Counter


def _normalize_answer(s: str) -> str:
    """HotpotQA's normalization: lowercase, strip punctuation, articles, extra ws."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
