"""Unused pieces from the strategy note.

PCA 128, num_leaves 15, the other recommended tree corner, unit-conversion
features, group-centered residual PCA, and a listwise MLP. Early stopping
uses the seed-0 training holdout. Validation is scored after every recipe is
refit. Test labels are not read.

A replacement is worth a later test run only when validation disagreement
reaches 293, or overall reaches 736 with disagreement still at least 289.
That is four questions above the frozen blend (289/355 and 732/798).
"""
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, read_jsonl
from linear_model import load_bundle
from listwise_mlp import fit_mlp, predict_mlp, predict_scores, refit_mlp, save_mlp
from rank_features import (
    _question_z,
    fit_embed_stats,
    labels_from,
    load_stats,
    matrix_for,
    mixed_mask,
    parsed_table,
    predict_rank,
    save_stats,
    unit_block,
)
from train_lgbm import build_features, split_questions, train_one

OUT = Path('runs/unused_search')
BAR_HARD = 293
BAR_ALL = 736
CURRENT_HARD = 289


def hits(labels, mixed, choice):
    all_ok = hard_ok = 0
    for i, picked in enumerate(choice):
        hit = int(labels[i, picked] == 1)
        all_ok += hit
        hard_ok += int(hit and mixed[i])
    return all_ok, hard_ok


def choice_from_scores(scores, valid):
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~valid] = -1e9
    return masked.argmax(axis=1)


def pairwise_scores(embeddings, checkpoint):
    params = load_bundle(checkpoint)
    scale = np.where(params['std'] == 0, 1.0, params['std'])
    values = np.asarray(embeddings, dtype=np.float64)
    if params.get('center'):
        values = values - values.mean(axis=1, keepdims=True)
    return (values - params['mean']) / scale @ params['weight'] + params['bias']


def sweep(left, right, labels, mixed, valid):
    zl, zr = _question_z(left), _question_z(right)
    best = None
    rows = []
    for alpha in np.linspace(0, 1, 21):
        choice = choice_from_scores(alpha * zl + (1 - alpha) * zr, valid)
        all_ok, hard_ok = hits(labels, mixed, choice)
        rows.append((float(alpha), all_ok, hard_ok))
        if best is None or (hard_ok, all_ok) > (best[1], best[2]):
            best = (float(alpha), hard_ok, all_ok)
    return rows, best


def train_rankers(bank, embeddings, labels, mixed, valid, body, dev):
    recipes = [
        {'name': 'pca128', 'mode': 'full', 'pca_dim': 128, 'residual_dim': 0,
         'rounds': 400, 'trees': {'num_leaves': 31}},
        {'name': 'leaves15', 'mode': 'full', 'pca_dim': 64, 'residual_dim': 0,
         'rounds': 400, 'trees': {'num_leaves': 15}},
        {'name': 'corner', 'mode': 'full', 'pca_dim': 64, 'residual_dim': 0,
         'rounds': 500, 'trees': {
             'num_leaves': 15, 'learning_rate': 0.03, 'max_depth': 4, 'min_child_samples': 50,
         }},
        {'name': 'units', 'mode': 'units', 'pca_dim': 64, 'residual_dim': 0,
         'rounds': 400, 'trees': {'num_leaves': 31}},
        {'name': 'resid32', 'mode': 'full', 'pca_dim': 64, 'residual_dim': 32,
         'rounds': 400, 'trees': {'num_leaves': 31}},
    ]
    weight = np.ones((len(bank), 4), dtype=np.float64)
    summaries = []
    for recipe in recipes:
        name = recipe['name']
        print(f'=== early stop {name}', flush=True)
        body_stats = fit_embed_stats(
            embeddings[body], labels[body], recipe['pca_dim'], recipe['residual_dim'],
        )
        features, _ = build_features(bank, embeddings, labels, recipe['mode'], body_stats)
        booster = train_one(
            features, labels, body, dev, mixed, 'lambdarank', weight, recipe['rounds'],
            {'dev_valid': valid[dev], 'dev_mixed': mixed[dev]}, recipe['trees'],
        )
        rounds = int(booster.best_iteration or booster.current_iteration())
        dev_score = float(next(iter(next(iter(booster.best_score.values())).values())))
        print(f'{name} rounds={rounds} dev_hard={dev_score:.6f} scores={booster.best_score}', flush=True)
        print(f'=== refit {name}', flush=True)
        full_stats = fit_embed_stats(embeddings, labels, recipe['pca_dim'], recipe['residual_dim'])
        features, _ = build_features(bank, embeddings, labels, recipe['mode'], full_stats)
        booster = train_one(
            features, labels, np.arange(len(bank)), dev, mixed, 'lambdarank', weight,
            rounds, None, recipe['trees'],
        )
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(directory / 'model.txt'))
        save_stats(directory, full_stats)
        config = {
            'feature_mode': recipe['mode'],
            'objective': 'lambdarank',
            'mixed_weight': 1.0,
            'rounds': rounds,
            'pca_dim': recipe['pca_dim'],
            'residual_dim': recipe['residual_dim'],
            'trees': recipe['trees'],
            'dev_hard_acc': dev_score,
        }
        (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        summaries.append(config | {'name': name})
    return summaries


def score_rankers(bank, arrays, labels, mixed, summaries):
    rows = []
    for spec in summaries:
        choice = predict_rank(bank, arrays, OUT / spec['name'])
        all_ok, hard_ok = hits(labels, mixed, choice)
        print(f"VAL {spec['name']} all {all_ok} hard {hard_ok}", flush=True)
        rows.append({
            'name': spec['name'], 'all_ok': all_ok, 'hard_ok': hard_ok,
            'dev_hard_acc': spec['dev_hard_acc'], 'rounds': spec['rounds'],
        })
    return rows


def overlap(labels, mixed, left, right):
    both = only_left = only_right = neither = 0
    for i in np.flatnonzero(mixed):
        a = labels[i, left[i]] == 1
        b = labels[i, right[i]] == 1
        both += int(a and b)
        only_left += int(a and not b)
        only_right += int(b and not a)
        neither += int(not a and not b)
    return {
        'both': both, 'only_left': only_left, 'only_right': only_right, 'neither': neither,
    }


def main():
    print(
        f'adopt only if hard>={BAR_HARD} or (all>={BAR_ALL} and hard>={CURRENT_HARD})',
        flush=True,
    )
    bank, arrays = load_inputs('train')
    gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    valid = np.array([[answer is not None for answer in row] for row in parsed])
    body, dev = split_questions(len(bank))
    coverage = unit_block(bank).reshape(-1, 9)
    fired = (coverage[:, [0, 1, 3, 4, 6, 7]].sum(axis=1) > 0).mean()
    print(f'unit feature fires on {fired:.4f} of train candidates', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    names = ('pca128', 'leaves15', 'corner', 'units', 'resid32')
    if all((OUT / name / 'model.txt').is_file() for name in names):
        summaries = []
        for name in names:
            config = json.loads((OUT / name / 'config.json').read_text(encoding='utf-8'))
            summaries.append(config | {'name': name})
        print('reusing saved rankers', flush=True)
    else:
        summaries = train_rankers(bank, arrays['embeddings'], labels, mixed, valid, body, dev)

    print('=== mlp early stop', flush=True)
    _, _, epochs, history = fit_mlp(bank, arrays['embeddings'], labels, body, dev, mixed)
    print(f'=== mlp refit epochs={epochs}', flush=True)
    params, stats = refit_mlp(bank, arrays['embeddings'], labels, mixed, epochs)
    save_mlp(OUT / 'listwise_mlp', params, stats, epochs)
    (OUT / 'mlp_dev.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')

    vbank, varrays = load_inputs('validation')
    vgold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    vparsed = parsed_table(vbank)
    vlabels = labels_from(vgold, vparsed)
    vmixed = mixed_mask(vparsed)
    vvalid = np.array([[answer is not None for answer in row] for row in vparsed])
    ranked = score_rankers(vbank, varrays, vlabels, vmixed, summaries)
    mlp_choice = predict_mlp(vbank, varrays, OUT / 'listwise_mlp')
    mlp_all, mlp_hard = hits(vlabels, vmixed, mlp_choice)
    print(f'VAL listwise_mlp all {mlp_all} hard {mlp_hard}', flush=True)

    blend_choice = json.loads(Path('runs/validation-blend.json').read_text(encoding='utf-8'))['selected']
    blend_all, blend_hard = hits(vlabels, vmixed, blend_choice)
    print(f'VAL current blend all {blend_all} hard {blend_hard}', flush=True)
    print('overlap mlp vs blend on disagreement', overlap(vlabels, vmixed, mlp_choice, blend_choice), flush=True)

    best = max(ranked, key=lambda row: (row['hard_ok'], row['all_ok']))
    print('best new ranker', best, flush=True)
    print(
        'overlap best ranker vs blend',
        overlap(vlabels, vmixed, predict_rank(vbank, varrays, OUT / best['name']), blend_choice),
        flush=True,
    )

    pair = pairwise_scores(varrays['embeddings'], 'runs/model_bundle_generalize')
    rank_features, _, _ = matrix_for(
        vbank, varrays['embeddings'],
        load_stats(OUT / best['name']),
        json.loads((OUT / best['name'] / 'config.json').read_text(encoding='utf-8'))['feature_mode'],
    )
    import lightgbm as lgb
    rank_scores = lgb.Booster(model_file=str(OUT / best['name'] / 'model.txt')).predict(
        rank_features, num_threads=4,
    ).reshape(len(vbank), 4)
    rank_rows, rank_best = sweep(rank_scores, pair, vlabels, vmixed, vvalid)
    print('sweep best-ranker + pairwise', rank_best, flush=True)

    mlp_scores = predict_scores(
        vbank, varrays['embeddings'], params, stats,
    )
    bcfg = json.loads(Path('runs/model_bundle_blend/config.json').read_text(encoding='utf-8'))
    bfeat, _, _ = matrix_for(vbank, varrays['embeddings'], load_stats('runs/model_bundle_blend'), bcfg['feature_mode'])
    b_rank = lgb.Booster(model_file='runs/model_bundle_blend/model.txt').predict(bfeat, num_threads=4)
    b_rank = np.asarray(b_rank, dtype=np.float64).reshape(len(vbank), 4)
    b_pair = pairwise_scores(varrays['embeddings'], 'runs/model_bundle_blend')
    blend_scores = float(bcfg['alpha']) * _question_z(b_rank) + (1 - float(bcfg['alpha'])) * _question_z(b_pair)
    mlp_rows, mlp_best = sweep(mlp_scores, blend_scores, vlabels, vmixed, vvalid)
    print('sweep mlp + current blend', mlp_best, flush=True)

    report = {
        'bar_hard': BAR_HARD,
        'bar_all': BAR_ALL,
        'rankers': ranked,
        'mlp': {'all_ok': mlp_all, 'hard_ok': mlp_hard, 'epochs': epochs},
        'blend': {'all_ok': blend_all, 'hard_ok': blend_hard},
        'ranker_pairwise_sweep_best': {'alpha': rank_best[0], 'hard_ok': rank_best[1], 'all_ok': rank_best[2]},
        'ranker_pairwise_sweep': [{'alpha': a, 'all_ok': b, 'hard_ok': c} for a, b, c in rank_rows],
        'mlp_blend_sweep_best': {'alpha': mlp_best[0], 'hard_ok': mlp_best[1], 'all_ok': mlp_best[2]},
        'mlp_blend_sweep': [{'alpha': a, 'all_ok': b, 'hard_ok': c} for a, b, c in mlp_rows],
    }
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('wrote', OUT / 'validation.json', flush=True)


if __name__ == '__main__':
    main()
