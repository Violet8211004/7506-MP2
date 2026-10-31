"""Retune the linear score for questions that look like the test bank.

The test bank has at least two different parsed answers in every group.
About half of train and validation do not. This script never opens
validation or test until a training-dev rule has already picked one recipe.

Selection metric: accuracy on development questions with at least two
distinct parsed numbers. The development slice is the seed-0 holdout of
475 training questions. A recipe is eligible only if that accuracy is at
least 2 percentage points above the E1 recipe trained on the same body.
Validation is then scored once. The recipe is kept only if it gets at
least two more of those disagreement questions right than the current
checkpoint.
"""
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_model import choose, load_bundle

LR = 0.2
MAX_STEPS = 160
DEV_GAIN_PP = 2.0
VAL_GAIN_QUESTIONS = 2
L2_GRID = (1e-2, 1e-1, 1.0)


def split_questions(n):
    order = np.random.default_rng(0).permutation(n)
    n_dev = n // 10
    return order[n_dev:], order[:n_dev]


def parsed_table(bank):
    table = []
    for row in bank:
        table.append([parse_answer(candidate['text']) for candidate in row['candidates']])
    return table


def labels_from(bank, gold, parsed):
    labels = np.empty((len(bank), 4), dtype=np.float64)
    for i, reference in enumerate(gold):
        gold_answer = parse_answer(reference['answer'])
        for j, answer in enumerate(parsed[i]):
            labels[i, j] = float(answer is not None and answer == gold_answer)
    return labels


def margins_and_mixed(parsed):
    mixed = np.empty(len(parsed), dtype=bool)
    margin = np.empty(len(parsed), dtype=np.int32)
    for i, answers in enumerate(parsed):
        counts = {}
        for answer in answers:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        mixed[i] = len(counts) >= 2
        if not counts:
            margin[i] = 0
        else:
            ordered = sorted(counts.values(), reverse=True)
            second = ordered[1] if len(ordered) > 1 else 0
            margin[i] = ordered[0] - second
    return mixed, margin


def question_center(embeddings):
    return embeddings - embeddings.mean(axis=1, keepdims=True)


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def choice_correct(scores, valid, hit, subset):
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~valid] = -np.inf
    choices = np.argmax(masked, axis=1)
    idx = np.flatnonzero(subset)
    ok = sum(int(hit[i, choices[i]]) for i in idx)
    return ok, int(idx.size)


def fit_pointwise(embeddings, labels, dev_embeddings, dev_valid, dev_hit, dev_mixed, l2):
    x = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float64)
    y = labels.reshape(-1).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (x - mean) / std
    dev_z = (dev_embeddings.astype(np.float64) - mean) / std
    n = z.shape[0]
    weight = np.zeros(z.shape[1], dtype=np.float64)
    bias = np.float64(0.0)
    best_ok = -1
    best = None
    at_80 = None
    for step in range(1, MAX_STEPS + 1):
        error = (sigmoid(z @ weight + bias) - y) / n
        weight -= LR * (z.T @ error + l2 * weight)
        bias -= LR * float(error.sum())
        ok, total = choice_correct(dev_z @ weight + bias, dev_valid, dev_hit, dev_mixed)
        if ok > best_ok:
            best_ok = ok
            best = (ok, total, step, weight.copy(), float(bias), mean.copy(), std.copy())
        if step == 80:
            at_80 = (ok, total)
    return best, at_80


def fit_pairwise(embeddings, labels, dev_embeddings, dev_valid, dev_hit, dev_mixed, l2):
    x = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (embeddings.astype(np.float64) - mean) / std
    diffs = []
    for zi, yi in zip(z, labels):
        for a in np.flatnonzero(yi == 1):
            for b in np.flatnonzero(yi == 0):
                diffs.append(zi[a] - zi[b])
    diffs = np.asarray(diffs, dtype=np.float64)
    n_pairs = diffs.shape[0]
    weight = np.zeros(z.shape[-1], dtype=np.float64)
    dev_z = (dev_embeddings.astype(np.float64) - mean) / std
    best_ok = -1
    best = None
    for step in range(1, MAX_STEPS + 1):
        coeff = (sigmoid(diffs @ weight) - 1.0) / n_pairs
        weight -= LR * (diffs.T @ coeff + l2 * weight)
        ok, total = choice_correct(dev_z @ weight, dev_valid, dev_hit, dev_mixed)
        if ok > best_ok:
            best_ok = ok
            best = (ok, total, step, weight.copy(), 0.0, mean.copy(), std.copy())
    return best


def refit_pointwise(embeddings, labels, steps, l2):
    x = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float64)
    y = labels.reshape(-1).astype(np.float64)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    z = (x - mean) / std
    n = z.shape[0]
    weight = np.zeros(z.shape[1], dtype=np.float64)
    bias = np.float64(0.0)
    for _ in range(steps):
        error = (sigmoid(z @ weight + bias) - y) / n
        weight -= LR * (z.T @ error + l2 * weight)
        bias -= LR * float(error.sum())
    return weight, float(bias), mean, std


def refit_pairwise(embeddings, labels, steps, l2):
    x = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float64)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    z = (embeddings.astype(np.float64) - mean) / std
    diffs = []
    for zi, yi in zip(z, labels):
        for a in np.flatnonzero(yi == 1):
            for b in np.flatnonzero(yi == 0):
                diffs.append(zi[a] - zi[b])
    diffs = np.asarray(diffs, dtype=np.float64)
    weight = np.zeros(z.shape[-1], dtype=np.float64)
    n_pairs = diffs.shape[0]
    for _ in range(steps):
        coeff = (sigmoid(diffs @ weight) - 1.0) / n_pairs
        weight -= LR * (diffs.T @ coeff + l2 * weight)
    return weight, 0.0, mean, std


def prepare_train():
    bank, arrays = load_inputs('train')
    gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if [row['id'] for row in gold] != [row['id'] for row in bank]:
        raise ValueError('Training references are not aligned')
    parsed = parsed_table(bank)
    labels = labels_from(bank, gold, parsed)
    mixed, margin = margins_and_mixed(parsed)
    body, dev = split_questions(len(bank))
    valid = np.array([[answer is not None for answer in row] for row in parsed])
    print(
        f'train={len(bank)} body={len(body)} dev={len(dev)} '
        f'mixed={int(mixed.sum())} unanimous={int((~mixed).sum())}',
        flush=True,
    )
    pos_mix = int(labels[mixed].sum())
    pos_uni = int(labels[~mixed].sum())
    print(f'positive labels mixed={pos_mix} unanimous={pos_uni}', flush=True)
    return {
        'embeddings': arrays['embeddings'],
        'labels': labels,
        'mixed': mixed,
        'margin': margin,
        'valid': valid,
        'body': body,
        'dev': dev,
        'hit': labels.astype(bool),
    }


def search(data):
    body, dev = data['body'], data['dev']
    emb = data['embeddings']
    labels = data['labels']
    dev_emb = emb[dev]
    dev_valid = data['valid'][dev]
    dev_hit = data['hit'][dev]
    dev_mixed = data['mixed'][dev]
    dev_low = data['margin'][dev] <= 1
    specs = []
    for name, question_mask, center in (
        ('all', np.ones(len(labels), dtype=bool), False),
        ('mixed', data['mixed'], False),
        ('mixed_center', data['mixed'], True),
    ):
        for l2 in L2_GRID:
            specs.append((name, 'pointwise', question_mask, center, l2))
    for l2 in (1e-2, 1e-1):
        specs.append(('mixed', 'pairwise', data['mixed'], False, l2))

    rows = []
    baseline = None
    for name, kind, question_mask, center, l2 in specs:
        use = question_mask.copy()
        use[dev] = False
        train_emb = emb[use]
        train_labels = labels[use]
        if center:
            train_emb = question_center(train_emb)
            scored_dev = question_center(dev_emb)
        else:
            scored_dev = dev_emb
        if kind == 'pointwise':
            best, at_80 = fit_pointwise(
                train_emb, train_labels, scored_dev, dev_valid, dev_hit, dev_mixed, l2,
            )
        else:
            best = fit_pairwise(
                train_emb, train_labels, scored_dev, dev_valid, dev_hit, dev_mixed, l2,
            )
            at_80 = None
        ok, total, step, weight, bias, mean, std = best
        low_scores = ((scored_dev.astype(np.float64) - mean) / std) @ weight + bias
        low_ok, low_n = choice_correct(low_scores, dev_valid, dev_hit, dev_low)
        row = {
            'data': name,
            'kind': kind,
            'center': center,
            'l2': l2,
            'step': step,
            'dev_ok': ok,
            'dev_n': total,
            'dev_pp': 100.0 * ok / total,
            'low_ok': low_ok,
            'low_n': low_n,
            'weight': weight,
            'bias': bias,
            'mean': mean,
            'std': std,
        }
        rows.append(row)
        print(
            f"{name:13} {kind:9} l2={l2:<6} step={step:<3} "
            f"dev {ok}/{total} = {row['dev_pp']:.2f}  low {low_ok}/{low_n}",
            flush=True,
        )
        if name == 'all' and kind == 'pointwise' and l2 == 1e-2 and at_80 is not None:
            baseline = {'ok': at_80[0], 'n': at_80[1], 'pp': 100.0 * at_80[0] / at_80[1]}
            print(f"baseline E1 recipe at step 80: {at_80[0]}/{at_80[1]} = {baseline['pp']:.2f}", flush=True)

    if baseline is None:
        raise RuntimeError('Missing E1 baseline')
    eligible = [row for row in rows if row['dev_pp'] >= baseline['pp'] + DEV_GAIN_PP]
    eligible.sort(key=lambda row: (row['dev_ok'], row['low_ok'], row['l2'], -row['step']), reverse=True)
    choice = {
        'baseline_dev_ok': baseline['ok'],
        'baseline_dev_n': baseline['n'],
        'baseline_dev_pp': baseline['pp'],
        'adopt_dev': bool(eligible),
    }
    if eligible:
        winner = eligible[0]
        choice.update({
            'data': winner['data'],
            'kind': winner['kind'],
            'center': winner['center'],
            'l2': winner['l2'],
            'step': winner['step'],
            'dev_ok': winner['dev_ok'],
            'dev_n': winner['dev_n'],
            'dev_pp': winner['dev_pp'],
        })
    out = Path('runs')
    out.mkdir(parents=True, exist_ok=True)
    lines = ['data,kind,center,l2,step,dev_ok,dev_n,dev_pp,low_ok,low_n']
    for row in rows:
        lines.append(','.join(str(row[key]) for key in (
            'data', 'kind', 'center', 'l2', 'step', 'dev_ok', 'dev_n', 'dev_pp', 'low_ok', 'low_n',
        )))
    (out / 'generalize_dev.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'generalize_choice.json').write_text(json.dumps(choice, indent=2) + '\n', encoding='utf-8')
    print('choice', json.dumps(choice), flush=True)
    return choice


def confirm(data, choice):
    if not choice['adopt_dev']:
        print('dev gate not met; validation was not opened', flush=True)
        return None
    use = np.ones(len(data['labels']), dtype=bool)
    if choice['data'] != 'all':
        use = data['mixed'].copy()
    emb = data['embeddings'][use]
    labels = data['labels'][use]
    center = choice['center']
    if center:
        emb = question_center(emb)
    if choice['kind'] == 'pointwise':
        weight, bias, mean, std = refit_pointwise(emb, labels, choice['step'], choice['l2'])
    else:
        weight, bias, mean, std = refit_pairwise(emb, labels, choice['step'], choice['l2'])
    params = {
        'weight': weight.astype(np.float32),
        'bias': np.float32(bias),
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
        'center': center,
    }
    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    parsed = parsed_table(bank)
    mixed, _ = margins_and_mixed(parsed)
    labels = labels_from(bank, gold, parsed)
    new_choices = score_with(bank, arrays['embeddings'], params)
    old_choices = choose(bank, arrays['embeddings'], load_bundle('runs/model_bundle'))
    new_ok = subset_hits(new_choices, labels, mixed)
    old_ok = subset_hits(old_choices, labels, mixed)
    new_all = subset_hits(new_choices, labels, np.ones(len(bank), dtype=bool))
    old_all = subset_hits(old_choices, labels, np.ones(len(bank), dtype=bool))
    result = {
        'new_dis_ok': new_ok,
        'old_dis_ok': old_ok,
        'dis_n': int(mixed.sum()),
        'new_all_ok': new_all,
        'old_all_ok': old_all,
        'n': len(bank),
        'adopt_val': new_ok >= old_ok + VAL_GAIN_QUESTIONS,
    }
    print(
        f"val disagreement new {new_ok}/{result['dis_n']} old {old_ok}/{result['dis_n']} "
        f"overall new {new_all}/{len(bank)} old {old_all}/{len(bank)} adopt={result['adopt_val']}",
        flush=True,
    )
    Path('runs/generalize_val.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if result['adopt_val']:
        directory = Path('runs/model_bundle_generalize')
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / 'linear.npz',
            weight=params['weight'],
            bias=np.asarray(params['bias'], dtype=np.float32),
            mean=params['mean'],
            std=params['std'],
            center=np.asarray(center),
        )
        config = {
            'method': choice['kind'],
            'data': choice['data'],
            'center': center,
            'steps': choice['step'],
            'learning_rate': LR,
            'l2': choice['l2'],
            'selected_on': 'train-dev disagreement accuracy, seed 0',
        }
        (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        print(f'saved {directory}', flush=True)
    return result


def score_with(bank, embeddings, params):
    from linear_model import parsed_mask
    values = np.asarray(embeddings, dtype=np.float64)
    if params['center']:
        values = question_center(values)
    std = np.where(params['std'] == 0, 1.0, params['std'])
    scores = (values - params['mean']) / std @ params['weight'].astype(np.float64) + float(params['bias'])
    scores = np.array(scores, copy=True)
    scores[~parsed_mask(bank)] = -np.inf
    return [int(i) for i in np.argmax(scores, axis=1)]


def subset_hits(choices, labels, subset):
    ok = 0
    for i in np.flatnonzero(subset):
        ok += int(labels[i, choices[i]] == 1)
    return ok


def main():
    data = prepare_train()
    choice = search(data)
    confirm(data, choice)


if __name__ == '__main__':
    main()
