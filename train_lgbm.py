"""Train LightGBM rankers and score validation once.

Test labels are not read. Early stopping uses a seed-0 holdout of the
training questions. The kept round count is then refit on the full training
split. Validation chooses among the finished recipes.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from common import DATA, load_inputs, read_jsonl
from rank_features import (
    fit_embed_stats,
    labels_from,
    matrix_for,
    mixed_mask,
    parsed_table,
    predict_rank,
    save_stats,
)

MODES = ('vote_text', 'embed', 'full')
OUT = Path('runs/lgbm_search')


def split_questions(n):
    order = np.random.default_rng(0).permutation(n)
    n_dev = n // 10
    return order[n_dev:], order[:n_dev]


def selection_metric(valid, mixed):
    def feval(preds, dataset):
        scores = preds.reshape(-1, 4).copy()
        scores[~valid] = -1e9
        labels = dataset.get_label().reshape(-1, 4)
        choice = scores.argmax(axis=1)
        idx = np.flatnonzero(mixed)
        hit = labels[idx, choice[idx]].mean() if len(idx) else 0.0
        return 'hard_acc', float(hit), True
    return feval


def params_for(objective):
    base = {
        'objective': objective,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_samples': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'seed': 42,
        'bagging_seed': 42,
        'feature_fraction_seed': 42,
        'verbosity': -1,
        'num_threads': 4,
        'metric': 'None',
        'force_col_wise': True,
    }
    if objective == 'lambdarank':
        base['label_gain'] = [0, 1]
    return base


def train_one(features, labels, body, dev, mixed, objective, row_weight, rounds, valid_sets, tree_params=None):
    x = features.reshape(len(labels) * 4, -1)
    y = labels.reshape(-1)
    train_idx = (body[:, None] * 4 + np.arange(4)).ravel()
    dtrain_kwargs = {
        'data': x[train_idx],
        'label': y[train_idx],
        'weight': row_weight.reshape(-1)[train_idx],
        'free_raw_data': False,
    }
    if objective == 'lambdarank':
        dtrain_kwargs['group'] = np.full(len(body), 4, dtype=np.int32)
    dtrain = lgb.Dataset(**dtrain_kwargs)
    callbacks = [lgb.log_evaluation(0)]
    eval_set = None
    feval = None
    if valid_sets:
        dev_idx = (dev[:, None] * 4 + np.arange(4)).ravel()
        dvalid_kwargs = {
            'data': x[dev_idx],
            'label': y[dev_idx],
            'reference': dtrain,
            'free_raw_data': False,
        }
        if objective == 'lambdarank':
            dvalid_kwargs['group'] = np.full(len(dev), 4, dtype=np.int32)
        dvalid = lgb.Dataset(**dvalid_kwargs)
        dev_valid = np.ones((len(dev), 4), dtype=bool)
        # Unparsed rows stay eligible here; inference masks them. The dev
        # metric below uses the same mask as inference.
        eval_set = [dvalid]
        feval = selection_metric(valid_sets['dev_valid'], valid_sets['dev_mixed'])
        callbacks = [
            lgb.early_stopping(40),
            lgb.log_evaluation(50),
        ]
    params = params_for(objective)
    if tree_params:
        params.update(tree_params)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=eval_set,
        feval=feval,
        callbacks=callbacks,
    )
    return booster


def build_features(bank, embeddings, labels, mode, stats):
    flat, _, parsed = matrix_for(bank, embeddings, stats, mode)
    return flat.reshape(len(bank), 4, -1), parsed


def run():
    bank, arrays = load_inputs('train')
    gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    valid = np.array([[answer is not None for answer in row] for row in parsed])
    body, dev = split_questions(len(bank))
    print(f'train={len(bank)} body={len(body)} dev={len(dev)} mixed={int(mixed.sum())}', flush=True)

    specs = [
        ('vote_text', 'lambdarank', 1.0),
        ('embed', 'lambdarank', 1.0),
        ('full', 'lambdarank', 1.0),
        ('full', 'lambdarank', 4.0),
        ('full', 'binary', 1.0),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for mode, objective, mixed_weight in specs:
        name = f'{mode}_{objective}_w{mixed_weight:g}'
        print(f'=== early stop {name}', flush=True)
        body_stats = fit_embed_stats(arrays['embeddings'][body], labels[body]) if mode != 'vote_text' else None
        features, _ = build_features(bank, arrays['embeddings'], labels, mode, body_stats)
        weight = np.ones(len(bank), dtype=np.float64)
        weight[mixed] = mixed_weight
        weight = np.repeat(weight[:, None], 4, axis=1)
        booster = train_one(
            features, labels, body, dev, mixed, objective, weight, 400,
            {'dev_valid': valid[dev], 'dev_mixed': mixed[dev]},
        )
        rounds = int(booster.best_iteration or booster.current_iteration())
        print(f'{name} rounds={rounds}', flush=True)
        print(f'=== refit {name}', flush=True)
        full_stats = fit_embed_stats(arrays['embeddings'], labels) if mode != 'vote_text' else {'note': 'none'}
        features, _ = build_features(bank, arrays['embeddings'], labels, mode, full_stats)
        booster = train_one(
            features, labels, np.arange(len(bank)), dev, mixed, objective, weight, rounds, None,
        )
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(directory / 'model.txt'))
        if mode != 'vote_text':
            save_stats(directory, full_stats)
        config = {
            'feature_mode': mode,
            'objective': objective,
            'mixed_weight': mixed_weight,
            'rounds': rounds,
            'pca_dim': 64,
        }
        (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        summaries.append(config | {'name': name, 'bytes': (directory / 'model.txt').stat().st_size})
        print(f'saved {directory}', flush=True)
    (OUT / 'search.json').write_text(json.dumps(summaries, indent=2) + '\n', encoding='utf-8')
    score_validation(summaries)


def score_validation(summaries):
    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    rows = ['name,all_ok,all_n,hard_ok,hard_n,asset_bytes']
    best = None
    for spec in summaries:
        choices = predict_rank(bank, arrays, OUT / spec['name'])
        all_ok = hard_ok = 0
        for i, choice in enumerate(choices):
            hit = int(labels[i, choice] == 1)
            all_ok += hit
            if mixed[i]:
                hard_ok += hit
        asset = sum(path.stat().st_size for path in (OUT / spec['name']).iterdir())
        rows.append(f"{spec['name']},{all_ok},{len(bank)},{hard_ok},{int(mixed.sum())},{asset}")
        print(
            f"VAL {spec['name']} all {all_ok}/{len(bank)} hard {hard_ok}/{int(mixed.sum())}",
            flush=True,
        )
        if best is None or (hard_ok, all_ok) > (best['hard_ok'], best['all_ok']):
            best = {'name': spec['name'], 'hard_ok': hard_ok, 'all_ok': all_ok}
    (OUT / 'validation.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print('best', best, flush=True)


if __name__ == '__main__':
    run()
