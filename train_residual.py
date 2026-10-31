"""Upweight the training questions the frozen blend still misses.

The strategy note says to give more weight to questions whose candidates
disagree, and to check that choice on validation. Here the extra weight is
only on the 179 training questions the submitted blend gets wrong. Step
count is chosen on the seed-0 development slice. Validation is scored after
the refit. Test labels are not read.

Replace the submitted blend only when two neighboring mix weights reach
disagreement 302, or overall 745 with disagreement still at least 298.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from close_model import frozen_blend_scores
from common import DATA, load_inputs, read_jsonl
from rank_features import (
    _question_z,
    fit_embed_stats,
    labels_from,
    mixed_mask,
    parsed_table,
    predict_rank,
    save_stats,
)
from train_lgbm import build_features, split_questions, train_one

BAR_HARD = 302
BAR_ALL = 745
CURRENT_HARD = 298
BLEND = Path('runs/model_bundle_blend_mlp')
OUT = Path('runs/residual_search')
WEIGHTS = (4.0, 8.0)


def counts(labels, mixed, choice):
    hit = labels[np.arange(len(choice)), choice] == 1
    return int(hit.sum()), int(hit[mixed].sum())


def main():
    print(
        f'adopt only if two neighboring weights reach hard>={BAR_HARD} '
        f'or (all>={BAR_ALL} and hard>={CURRENT_HARD})',
        flush=True,
    )
    bank, arrays = load_inputs('train')
    gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    valid = np.array([[answer is not None for answer in row] for row in parsed])
    body, dev = split_questions(len(bank))
    print('scoring blend on train', flush=True)
    scores, score_valid = frozen_blend_scores(bank, arrays['embeddings'], BLEND)
    scores = scores.copy()
    scores[~score_valid] = -1e9
    missed = labels[np.arange(len(bank)), scores.argmax(1)] != 1
    print(f'missed {int(missed.sum())} of which dev {int(missed[dev].sum())}', flush=True)

    chosen = None
    for extra in WEIGHTS:
        print(f'=== early stop weight {extra}', flush=True)
        row_weight = np.ones((len(bank), 4), dtype=np.float64)
        row_weight[missed] = extra
        stats = fit_embed_stats(arrays['embeddings'][body], labels[body])
        features, _ = build_features(bank, arrays['embeddings'], labels, 'full', stats)
        booster = train_one(
            features, labels, body, dev, mixed, 'lambdarank', row_weight, 400,
            {'dev_valid': valid[dev], 'dev_mixed': mixed[dev]},
        )
        rounds = int(booster.best_iteration or booster.current_iteration())
        dev_acc = float(next(iter(next(iter(booster.best_score.values())).values())))
        dev_ok = int(round(dev_acc * int(mixed[dev].sum())))
        print(f'weight {extra} rounds {rounds} dev {dev_ok}/{int(mixed[dev].sum())}', flush=True)
        # Higher development accuracy, then the smaller extra weight.
        key = (dev_ok, -extra)
        if chosen is None or key > chosen[0]:
            chosen = (key, extra, rounds, dev_ok)

    extra, rounds, dev_ok = chosen[1], chosen[2], chosen[3]
    print(f'refit weight {extra} rounds {rounds}', flush=True)
    row_weight = np.ones((len(bank), 4), dtype=np.float64)
    row_weight[missed] = extra
    stats = fit_embed_stats(arrays['embeddings'], labels)
    features, _ = build_features(bank, arrays['embeddings'], labels, 'full', stats)
    booster = train_one(
        features, labels, np.arange(len(bank)), dev, mixed, 'lambdarank',
        row_weight, rounds, None,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    directory = OUT / 'ranker'
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / 'model.txt'))
    save_stats(directory, stats)
    config = {
        'feature_mode': 'full',
        'objective': 'lambdarank',
        'residual_weight': extra,
        'rounds': rounds,
        'dev_hard': dev_ok,
    }
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    vbank, varrays = load_inputs('validation')
    vgold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    vparsed = parsed_table(vbank)
    vlabels = labels_from(vgold, vparsed)
    vmixed = mixed_mask(vparsed)
    choice = predict_rank(vbank, varrays, directory)
    all_ok, hard_ok = counts(vlabels, vmixed, choice)
    print(f'val ranker all {all_ok} hard {hard_ok}', flush=True)
    base, base_valid = frozen_blend_scores(vbank, varrays['embeddings'], BLEND)
    import lightgbm as lgb
    from rank_features import load_stats, matrix_for
    feat, rank_valid, _ = matrix_for(vbank, varrays['embeddings'], load_stats(directory), 'full')
    rank = lgb.Booster(model_file=str(directory / 'model.txt')).predict(feat, num_threads=4)
    rank = np.asarray(rank, dtype=np.float64).reshape(len(vbank), 4)
    both = only_new = only_old = neither = 0
    base_choice = np.array(base, copy=True)
    base_choice[~base_valid] = -1e9
    base_choice = base_choice.argmax(1)
    for i in np.flatnonzero(vmixed):
        new_hit = vlabels[i, choice[i]] == 1
        old_hit = vlabels[i, base_choice[i]] == 1
        both += int(new_hit and old_hit)
        only_new += int(new_hit and not old_hit)
        only_old += int(old_hit and not new_hit)
        neither += int(not new_hit and not old_hit)
    print(f'overlap both {both} only_new {only_new} only_old {only_old} neither {neither}', flush=True)
    sweep = []
    for alpha in np.linspace(0, 1, 21):
        mixed_scores = alpha * _question_z(rank) + (1.0 - alpha) * _question_z(base)
        mixed_scores = np.array(mixed_scores, copy=True)
        mixed_scores[~rank_valid] = -1e9
        picked = mixed_scores.argmax(1)
        a_ok, h_ok = counts(vlabels, vmixed, picked)
        sweep.append({'alpha': float(alpha), 'all_ok': a_ok, 'hard_ok': h_ok})
        print(f'alpha {alpha:.2f} all {a_ok} hard {h_ok}', flush=True)
    meeting = [
        row for row in sweep
        if row['hard_ok'] >= BAR_HARD or (row['all_ok'] >= BAR_ALL and row['hard_ok'] >= CURRENT_HARD)
    ]
    alphas = [row['alpha'] for row in meeting]
    adjacent = any(abs(a - b) <= 0.051 and a != b for a in alphas for b in alphas)
    adopt = bool(meeting) and adjacent
    picked = (
        max(meeting, key=lambda row: (row['hard_ok'], row['all_ok'], -row['alpha']))
        if meeting else max(sweep, key=lambda row: (row['hard_ok'], row['all_ok'], -row['alpha']))
    )
    print(f'adopt {adopt} picked {picked}', flush=True)
    (OUT / 'result.json').write_text(json.dumps({
        'dev_hard': dev_ok,
        'residual_weight': extra,
        'rounds': rounds,
        'val_ranker': [all_ok, hard_ok],
        'overlap': {'both': both, 'only_new': only_new, 'only_old': only_old, 'neither': neither},
        'sweep': sweep,
        'adopt': adopt,
        'picked': picked,
    }, indent=2) + '\n', encoding='utf-8')
    if not adopt:
        print('keep current checkpoint', flush=True)
        return
    dest = Path('runs/model_bundle_residual')
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ('model.txt', 'preprocess.npz', 'linear.npz', 'mlp.npz'):
        shutil.copyfile(BLEND / name, dest / name)
    shutil.copyfile(directory / 'model.txt', dest / 'residual_model.txt')
    shutil.copyfile(directory / 'preprocess.npz', dest / 'residual_preprocess.npz')
    (dest / 'config.json').write_text(json.dumps({
        'method': 'blend_residual',
        'residual_alpha': picked['alpha'],
        'mlp_alpha': 0.25,
        'rank_alpha': 0.45,
        'feature_mode': 'full',
    }, indent=2) + '\n', encoding='utf-8')
    print('wrote', dest, flush=True)


if __name__ == '__main__':
    main()
