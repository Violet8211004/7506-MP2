"""Pairwise score for questions whose answers nearly tie.

The training pairs are a correct candidate against an incorrect one.
Features are the embedding difference, the vote-count difference, and the
difference in whether each answer is shared by another candidate. A symmetric
"these two texts have the same number" bit is constant on those pairs, because
two texts with the same parsed number have the same label.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from linear_model import load_bundle
from listwise_mlp import load_mlp, predict_scores
from rank_features import _question_z, load_stats, matrix_for


def agreement(parsed):
    """Vote count and whether that answer appears at least twice."""
    vote = np.zeros((len(parsed), 4), dtype=np.float64)
    for i, answers in enumerate(parsed):
        counts = {}
        for answer in answers:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        for j, answer in enumerate(answers):
            if answer is not None:
                vote[i, j] = counts[answer]
    return vote, (vote >= 2).astype(np.float64)


def close_mask(parsed):
    """At least two distinct numbers, and the lead is at most one vote."""
    mask = np.empty(len(parsed), dtype=bool)
    for i, answers in enumerate(parsed):
        counts = {}
        for answer in answers:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        if len(counts) < 2:
            mask[i] = False
            continue
        ordered = sorted(counts.values(), reverse=True)
        mask[i] = ordered[0] - ordered[1] <= 1
    return mask


def standardize_fit(embeddings, rows):
    flat = np.asarray(embeddings[rows], dtype=np.float64).reshape(-1, embeddings.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _z(embeddings, mean, std):
    return (np.asarray(embeddings, dtype=np.float64) - mean) / std


def pair_arrays(z, vote, match, labels, rows):
    diffs = []
    vote_gap = []
    match_gap = []
    for i in rows:
        yi = labels[i]
        positive = np.flatnonzero(yi == 1)
        negative = np.flatnonzero(yi == 0)
        for a in positive:
            for b in negative:
                diffs.append(z[i, a] - z[i, b])
                vote_gap.append(vote[i, a] - vote[i, b])
                match_gap.append(match[i, a] - match[i, b])
    return (
        np.asarray(diffs, dtype=np.float64),
        np.asarray(vote_gap, dtype=np.float64),
        np.asarray(match_gap, dtype=np.float64),
    )


def scores_from(z, vote, match, weight, vote_coef, match_coef):
    """Sum of directed pair scores. Equivalent to 4 * f_i - sum_k f_k."""
    linear = z @ weight
    vote_term = vote_coef * vote
    match_term = match_coef * match
    total = linear + vote_term + match_term
    return 4.0 * total - total.sum(axis=1, keepdims=True)


def logistic_fit(diffs, vote_gap, match_gap, dev_pack, steps, l2, lr=0.2):
    """Full-batch steps. dev_pack is (z, vote, match, valid, hit, mixed) or None."""
    weight = np.zeros(diffs.shape[1], dtype=np.float64)
    vote_coef = 0.0
    match_coef = 0.0
    n_pairs = diffs.shape[0]
    best = None
    for step in range(1, steps + 1):
        logit = diffs @ weight + vote_coef * vote_gap + match_coef * match_gap
        coeff = (1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))) - 1.0) / n_pairs
        weight -= lr * (diffs.T @ coeff + l2 * weight)
        vote_coef -= lr * (float(vote_gap @ coeff) + l2 * vote_coef)
        match_coef -= lr * (float(match_gap @ coeff) + l2 * match_coef)
        if dev_pack is not None:
            z, vote, match, valid, hit, mixed = dev_pack
            ok, total = correct_count(
                scores_from(z, vote, match, weight, vote_coef, match_coef),
                valid, hit, mixed,
            )
            if best is None or ok > best[0] or (ok == best[0] and step < best[1]):
                best = (ok, step, weight.copy(), vote_coef, match_coef, total)
    return best


def correct_count(scores, valid, hit, subset):
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~valid] = -np.inf
    choice = np.argmax(masked, axis=1)
    idx = np.flatnonzero(subset)
    ok = int(hit[idx, choice[idx]].sum()) if len(idx) else 0
    return ok, int(len(idx))


def frozen_blend_scores(bank, embeddings, checkpoint):
    """The submitted blend, before unparsed answers are removed."""
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    features, valid, _ = matrix_for(
        bank, embeddings, load_stats(checkpoint), config['feature_mode'],
    )
    rank = lgb.Booster(model_file=str(checkpoint / 'model.txt')).predict(features, num_threads=4)
    rank = np.asarray(rank, dtype=np.float64).reshape(len(bank), 4)
    linear = load_bundle(checkpoint)
    scale = np.where(linear['std'] == 0, 1.0, linear['std'])
    values = np.asarray(embeddings, dtype=np.float64)
    if linear.get('center'):
        values = values - values.mean(axis=1, keepdims=True)
    pair = (values - linear['mean']) / scale @ linear['weight'] + linear['bias']
    params, mlp_stats = load_mlp(checkpoint)
    mlp = np.asarray(predict_scores(bank, embeddings, params, mlp_stats), dtype=np.float64)
    rank_alpha = float(config.get('rank_alpha', 0.45))
    mlp_alpha = float(config.get('mlp_alpha', 0.25))
    blend = rank_alpha * _question_z(rank) + (1.0 - rank_alpha) * _question_z(pair)
    scores = mlp_alpha * _question_z(mlp) + (1.0 - mlp_alpha) * _question_z(blend)
    return np.asarray(scores, dtype=np.float64), valid


def load_close(checkpoint):
    with np.load(Path(checkpoint) / 'close.npz') as bundle:
        return {
            'weight': bundle['weight'].astype(np.float64),
            'vote_coef': float(bundle['vote_coef']),
            'match_coef': float(bundle['match_coef']),
            'mean': bundle['mean'].astype(np.float64),
            'std': bundle['std'].astype(np.float64),
        }


def close_scores(embeddings, parsed, params):
    z = _z(embeddings, params['mean'], params['std'])
    vote, match = agreement(parsed)
    return scores_from(z, vote, match, params['weight'], params['vote_coef'], params['match_coef'])


def predict_close_blend(bank, arrays, checkpoint):
    from rank_features import parsed_table

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    alpha = float(config['close_alpha'])
    base, valid = frozen_blend_scores(bank, arrays['embeddings'], checkpoint)
    extra = close_scores(arrays['embeddings'], parsed_table(bank), load_close(checkpoint))
    scores = alpha * _question_z(extra) + (1.0 - alpha) * _question_z(base)
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[~valid] = -np.inf
    return [int(i) for i in np.argmax(scores, axis=1)]
