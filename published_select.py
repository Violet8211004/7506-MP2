"""Published selectors for four already-written solutions.

Cobbe et al. (2021) rank candidates with a verifier and also let the
top-ranked ones vote. DiVeRSe (Li et al., 2023) sums verifier
probabilities that share a final answer. Self-consistency (Wang et al.,
2023) is minimum-Bayes-risk with exact answer match. Embedding MBR
picks the candidate most similar to the others, and the
structure-conditional form multiplies that similarity by answer match.

This file scores those selectors and their sums on a split. It does not
read the test labels.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from common import DATA, load_inputs, read_jsonl
from linear_model import load_bundle
from listwise_mlp import load_mlp, predict_scores
from rank_features import _question_z, labels_from, load_stats, matrix_for, mixed_mask, parsed_table

BLEND = Path('runs/model_bundle_blend_mlp')


def component_scores(bank, embeddings, checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    features, valid, parsed = matrix_for(
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
    blend = mlp_alpha * _question_z(mlp) + (1.0 - mlp_alpha) * _question_z(blend)
    return {
        'rank': rank,
        'pair': np.asarray(pair, dtype=np.float64),
        'mlp': mlp,
        'blend': np.asarray(blend, dtype=np.float64),
        'valid': valid,
        'parsed': parsed,
    }


def _sigmoid(scores):
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))


def answer_mass(strength, parsed, valid):
    """DiVeRSe: each candidate receives the total strength of its answer."""
    mass = np.zeros_like(strength, dtype=np.float64)
    for i, answers in enumerate(parsed):
        groups = {}
        for j, answer in enumerate(answers):
            if not valid[i, j]:
                continue
            groups[answer] = groups.get(answer, 0.0) + float(strength[i, j])
        for j, answer in enumerate(answers):
            if valid[i, j]:
                mass[i, j] = groups[answer]
    return mass


def cosine_matrix(embeddings, valid):
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    unit = values / np.clip(norms, 1e-8, None)
    sim = unit @ np.swapaxes(unit, 1, 2)
    sim = sim * valid[:, None, :].astype(np.float64)
    diagonal = np.arange(4)
    sim[:, diagonal, diagonal] = 0.0
    return sim


def mbr_cosine(embeddings, valid):
    sim = cosine_matrix(embeddings, valid)
    count = valid.sum(axis=1, keepdims=True) - valid.astype(np.float64)
    return sim.sum(axis=-1) / np.clip(count, 1.0, None)


def mbr_exact(parsed, valid):
    """Self-consistency / MBR with exact answer match, excluding self."""
    score = np.zeros((len(parsed), 4), dtype=np.float64)
    for i, answers in enumerate(parsed):
        for j, answer in enumerate(answers):
            if not valid[i, j]:
                continue
            score[i, j] = sum(
                valid[i, k] and answers[k] == answer
                for k in range(4) if k != j
            )
    return score


def mbr_structure(embeddings, parsed, valid):
    """Utility is answer match times cosine similarity."""
    sim = cosine_matrix(embeddings, valid)
    score = np.zeros((len(parsed), 4), dtype=np.float64)
    for i, answers in enumerate(parsed):
        for j, answer in enumerate(answers):
            if not valid[i, j]:
                continue
            total = 0.0
            for k in range(4):
                if k != j and valid[i, k] and answers[k] == answer:
                    total += sim[i, j, k]
            score[i, j] = total
    return score


def topk_vote(scores, parsed, valid, k):
    """Cobbe et al.: majority vote among the k highest verifier scores."""
    choice = np.zeros(len(parsed), dtype=np.int64)
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~valid] = -1e9
    order = np.argsort(-masked, axis=1)
    for i, answers in enumerate(parsed):
        picked = []
        for j in order[i]:
            if not valid[i, j]:
                continue
            picked.append(int(j))
            if len(picked) == k:
                break
        if not picked:
            choice[i] = 0
            continue
        counts = {}
        for j in picked:
            counts[answers[j]] = counts.get(answers[j], 0) + 1
        best = max(counts.values())
        winners = {answer for answer, count in counts.items() if count == best}
        chosen = None
        for j in picked:
            if answers[j] in winners and (chosen is None or masked[i, j] > masked[i, chosen]):
                chosen = j
        choice[i] = 0 if chosen is None else chosen
    return choice


def choose_scores(scores, valid):
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~valid] = -np.inf
    return np.argmax(masked, axis=1)


def fuse(parts, valid):
    total = np.zeros_like(parts[0], dtype=np.float64)
    for part in parts:
        total = total + _question_z(np.asarray(part, dtype=np.float64))
    return choose_scores(total, valid)


def evaluate(labels, mixed, choice):
    hit = labels[np.arange(len(choice)), choice] == 1
    return int(hit.sum()), int(hit[mixed].sum())


def selector_table(bank, embeddings, checkpoint):
    parts = component_scores(bank, embeddings, checkpoint)
    valid = parts['valid']
    parsed = parts['parsed']
    blend = parts['blend']
    prob = _sigmoid(blend)
    mass = answer_mass(prob, parsed, valid)
    # Mass decides the answer; the verifier breaks ties inside that answer.
    diverse = mass * 100.0 + blend
    cos = mbr_cosine(embeddings, valid)
    exact = mbr_exact(parsed, valid)
    struct = mbr_structure(embeddings, parsed, valid)
    named = {
        'bon_blend': blend,
        'diverse_blend': diverse,
        'mbr_cosine': cos,
        'mbr_exact': exact,
        'mbr_structure': struct,
        'diverse_rank': answer_mass(_sigmoid(parts['rank']), parsed, valid) * 100.0 + parts['rank'],
        'diverse_pair': answer_mass(_sigmoid(parts['pair']), parsed, valid) * 100.0 + parts['pair'],
        'diverse_mlp': answer_mass(_sigmoid(parts['mlp']), parsed, valid) * 100.0 + parts['mlp'],
    }
    choices = {name: choose_scores(score, valid) for name, score in named.items()}
    choices['cobbe_top2'] = topk_vote(blend, parsed, valid, 2)
    choices['cobbe_top3'] = topk_vote(blend, parsed, valid, 3)
    groups = {
        'bon+cosine': ['bon_blend', 'mbr_cosine'],
        'bon+structure': ['bon_blend', 'mbr_structure'],
        'bon+exact': ['bon_blend', 'mbr_exact'],
        'diverse+cosine': ['diverse_blend', 'mbr_cosine'],
        'diverse+structure': ['diverse_blend', 'mbr_structure'],
        'diverse+exact': ['diverse_blend', 'mbr_exact'],
        'bon+diverse': ['bon_blend', 'diverse_blend'],
        'bon+diverse+cosine': ['bon_blend', 'diverse_blend', 'mbr_cosine'],
        'bon+diverse+structure': ['bon_blend', 'diverse_blend', 'mbr_structure'],
        'bon+exact+cosine': ['bon_blend', 'mbr_exact', 'mbr_cosine'],
        'all_five': ['bon_blend', 'diverse_blend', 'mbr_cosine', 'mbr_exact', 'mbr_structure'],
        'three_verifiers': ['diverse_rank', 'diverse_pair', 'diverse_mlp'],
        'three_verifiers+bon': ['diverse_rank', 'diverse_pair', 'diverse_mlp', 'bon_blend'],
    }
    for name, keys in groups.items():
        choices[name] = fuse([named[key] for key in keys], valid)
    return choices, valid, parsed


def feature_matrix(bank, embeddings, checkpoint):
    """One row of published selector scores per candidate."""
    parts = component_scores(bank, embeddings, checkpoint)
    valid = parts['valid']
    parsed = parts['parsed']
    prob = _sigmoid(parts['blend'])
    columns = [
        parts['blend'],
        parts['rank'],
        parts['pair'],
        parts['mlp'],
        answer_mass(prob, parsed, valid),
        answer_mass(_sigmoid(parts['rank']), parsed, valid),
        answer_mass(_sigmoid(parts['pair']), parsed, valid),
        answer_mass(_sigmoid(parts['mlp']), parsed, valid),
        mbr_cosine(embeddings, valid),
        mbr_exact(parsed, valid),
        mbr_structure(embeddings, parsed, valid),
        step_weights(bank),
    ]
    features = np.stack(columns, axis=-1)
    return features, valid, parsed


def step_weights(bank):
    """Fu et al.: more reasoning steps get a larger self-consistency weight."""
    weight = np.zeros((len(bank), 4), dtype=np.float64)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            text = candidate['text']
            weight[i, j] = text.count('\n') + text.count('=') + 1.0
    return weight


def complexity_vote(bank, parsed, valid):
    weight = step_weights(bank)
    return choose_scores(answer_mass(weight, parsed, valid) * 100.0 + weight, valid)


def fit_linear_verifier(features, labels, steps=80, lr=0.2, l2=1e-2):
    """Cobbe-style solution verifier on the selector scores."""
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    y = labels.reshape(-1).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (x - mean) / std
    weight = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    n = z.shape[0]
    for _ in range(steps):
        prob = 1.0 / (1.0 + np.exp(-np.clip(z @ weight + bias, -50, 50)))
        error = (prob - y) / n
        weight -= lr * (z.T @ error + l2 * weight)
        bias -= lr * float(error.sum())
    return weight, bias, mean, std


def apply_linear(features, weight, bias, mean, std):
    scale = np.where(std == 0, 1.0, std)
    z = (features - mean) / scale
    return z @ weight + bias


def committee(choices):
    """Each selector votes for an index. Ties keep the smallest index."""
    stacked = np.stack(list(choices), axis=0)
    out = np.zeros(stacked.shape[1], dtype=np.int64)
    for i in range(stacked.shape[1]):
        counts = np.bincount(stacked[:, i], minlength=4)
        out[i] = int(np.argmax(counts))
    return out


def score_split(split, checkpoint=BLEND):
    bank, arrays = load_inputs(split)
    gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    choices, _, _ = selector_table(bank, arrays['embeddings'], checkpoint)
    rows = []
    for name, choice in choices.items():
        all_ok, hard_ok = evaluate(labels, mixed, choice)
        rows.append((all_ok, hard_ok, name))
        print(f'{split} {name} all {all_ok} hard {hard_ok}', flush=True)
    rows.sort(reverse=True)
    print('best', rows[0], flush=True)
    return rows


def fit_pairwise_verifier(features, labels, mixed, steps=12, lr=0.2, l2=0.1):
    """Pairwise logistic on selector-score differences, mixed questions only."""
    x = features.astype(np.float64)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (x - mean) / std
    left, right = [], []
    for i in np.flatnonzero(mixed):
        good = [j for j in range(4) if labels[i, j] == 1]
        bad = [j for j in range(4) if labels[i, j] == 0]
        for g in good:
            for b in bad:
                left.append(z[i, g] - z[i, b])
                right.append(z[i, b] - z[i, g])
    diff = np.concatenate([np.stack(left), np.stack(right)], axis=0)
    target = np.concatenate([
        np.ones(len(left)),
        np.zeros(len(right)),
    ])
    weight = np.zeros(diff.shape[1], dtype=np.float64)
    n = diff.shape[0]
    for _ in range(steps):
        prob = 1.0 / (1.0 + np.exp(-np.clip(diff @ weight, -50, 50)))
        error = (prob - target) / n
        weight -= lr * (diff.T @ error + l2 * weight)
    return weight, mean, std


if __name__ == '__main__':
    train_bank, train_arrays = load_inputs('train')
    train_gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    train_parsed = parsed_table(train_bank)
    train_labels = labels_from(train_gold, train_parsed)
    train_mixed = mixed_mask(train_parsed)
    train_x, _, _ = feature_matrix(train_bank, train_arrays['embeddings'], BLEND)
    point = fit_linear_verifier(train_x, train_labels)
    pair = fit_pairwise_verifier(train_x, train_labels, train_mixed)

    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    choices, valid, _ = selector_table(bank, arrays['embeddings'], BLEND)
    choices['complexity_vote'] = complexity_vote(bank, parsed, valid)
    choices['committee_all'] = committee(list(choices.values()))
    val_x, val_valid, _ = feature_matrix(bank, arrays['embeddings'], BLEND)
    choices['stacked_verifier'] = choose_scores(apply_linear(val_x, *point), val_valid)
    choices['stacked_pairwise'] = choose_scores(apply_linear(val_x, pair[0], 0.0, pair[1], pair[2]), val_valid)
    rows = []
    for name, choice in choices.items():
        all_ok, hard_ok = evaluate(labels, mixed, choice)
        rows.append((all_ok, hard_ok, name))
        print(f'validation {name} all {all_ok} hard {hard_ok}', flush=True)
    rows.sort(reverse=True)
    print('best', rows[0], flush=True)
    print('current_blend', [row for row in rows if row[2] == 'bon_blend'][0], flush=True)
