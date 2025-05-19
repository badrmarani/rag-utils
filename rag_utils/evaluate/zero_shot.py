from typing import List, Optional

from thefuzz import fuzz, process


def drop_duplicates(files: List[str], threshold: int = 80) -> List[str]:
    out = files.copy()
    for i, x1 in enumerate(files, start=1):
        for x2 in files[i:]:
            s = fuzz.ratio(x1, x2)
            if s >= threshold and x1 in out:
                out.remove(x1)
                break
    return out


def num_intersection(Y_pred: List[str], Y_true: List[str], threshold: int = 80) -> int:
    count = 0
    for y_pred in drop_duplicates(Y_pred, threshold):
        ((_, s),) = process.extract(y_pred, drop_duplicates(Y_true, threshold), limit=1)
        if s >= threshold:
            count += 1
    return count


def num_union(Y_pred: List[str], Y_true: List[str], threshold: int = 80) -> int:
    return len(drop_duplicates(Y_pred + Y_true, threshold=threshold))


def accuracy_at_k(
    Y_pred: List[List[str]],
    Y_true: List[List[str]],
    k: Optional[int] = None,
    threshold: int = 80,
) -> float:
    score = 0.0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if len(y_pred) == 0 or len(y_true) == 0:
            continue
        y_pred, y_true = (
            drop_duplicates(y_pred, threshold),
            drop_duplicates(y_true, threshold),
        )
        score += num_intersection(y_pred[:k], y_true, threshold) / num_union(
            y_pred[:k], y_true, threshold
        )
    return score / len(Y_pred)


def precision_at_k(
    Y_pred: List[List[str]],
    Y_true: List[List[str]],
    k: Optional[int] = None,
    threshold: int = 80,
) -> float:
    score = 0.0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if len(y_pred) == 0 or len(y_true) == 0:
            continue
        y_pred, y_true = (
            drop_duplicates(y_pred, threshold),
            drop_duplicates(y_true, threshold),
        )
        score += num_intersection(y_pred[:k], y_true, threshold) / len(y_true)
    return score / len(Y_pred)


def recall_at_k(
    Y_pred: List[List[str]],
    Y_true: List[List[str]],
    k: Optional[int] = None,
    threshold: int = 80,
) -> float:
    score = 0.0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if len(y_pred) == 0 or len(y_true) == 0:
            continue
        y_pred, y_true = (
            drop_duplicates(y_pred, threshold),
            drop_duplicates(y_true, threshold),
        )
        score += num_intersection(y_pred[:k], y_true, threshold) / len(y_pred[:k])
    return score / len(Y_pred)


def hit_atleast_m_at_k(
    Y_pred: List[List[str]],
    Y_true: List[List[str]],
    m: int = 1,
    k: Optional[int] = None,
    threshold: int = 80,
) -> float:
    score = 0.0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if len(y_pred) == 0 or len(y_true) == 0:
            continue
        y_pred, y_true = (
            drop_duplicates(y_pred, threshold),
            drop_duplicates(y_true, threshold),
        )
        score += float((num_intersection(y_pred[:k], Y_true, threshold)) >= m)
    return score / len(Y_pred)
