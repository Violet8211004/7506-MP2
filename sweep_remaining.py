"""Finish the ensemble checks that were not run.

Direct three-way weights, and temperature used as a relative scale when
mixing scores. Both are chosen on validation. Test labels are not read.

Replace the frozen blend only if disagreement reaches 302, or overall
reaches 745 with disagreement still at least 298.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from common import DATA, load_inputs, read_jsonl
from linear_model import load_bundle
from listwise_mlp import load_mlp, predict_scores
from rank_features import _question_z, labels_from, load_stats, matrix_for, mixed_mask, parsed_table

BAR_HARD = 302
BAR_ALL = 745
CURRENT_HARD = 298
CKPT = Path('runs/model_bundle_blend_mlp')


def hits(labels, mixed, scores, valid):
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~valid] = -1e9
    choice = masked.argmax(axis=1)
    all_ok = int((labels[np.arange(len(choice)), choice] == 1).sum())
    idx = np.flatnonzero(mixed)
    hard_ok = int((labels[idx, choice[idx]] == 1).sum())
    return all_ok, hard_ok


def softmax(scores, temperature):
    scaled = scores / temperature
    shift = scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled - shift)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    print(
        f'adopt only if hard>={BAR_HARD} or (all>={BAR_ALL} and hard>={CURRENT_HARD})',
        flush=True,
    )
    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    labels = labels_from(gold, parsed_table(bank))
    mixed = mixed_mask(parsed_table(bank))
    valid = np.array([[answer is not None for answer in row] for row in parsed_table(bank)])
    config = json.loads((CKPT / 'config.json').read_text(encoding='utf-8'))
    features, _, _ = matrix_for(bank, arrays['embeddings'], load_stats(CKPT), config['feature_mode'])
    rank = lgb.Booster(model_file=str(CKPT / 'model.txt')).predict(features, num_threads=4)
    rank = np.asarray(rank, dtype=np.float64).reshape(len(bank), 4)
    linear = load_bundle(CKPT)
    scale = np.where(linear['std'] == 0, 1.0, linear['std'])
    values = np.asarray(arrays['embeddings'], dtype=np.float64)
    pair = (values - linear['mean']) / scale @ linear['weight'] + linear['bias']
    params, stats = load_mlp(CKPT)
    mlp = np.asarray(predict_scores(bank, arrays['embeddings'], params, stats), dtype=np.float64)
    zr, zp, zm = _question_z(rank), _question_z(pair), _question_z(mlp)
    nested = 0.45 * zr + 0.55 * zp
    current = 0.25 * zm + 0.75 * _question_z(nested)
    print('current nested', hits(labels, mixed, current, valid), flush=True)

    grid = np.linspace(0, 1, 21)
    best_linear = None
    for weight_rank in grid:
        for weight_pair in grid:
            weight_mlp = 1.0 - weight_rank - weight_pair
            if weight_mlp < -1e-9:
                continue
            scores = weight_rank * zr + weight_pair * zp + weight_mlp * zm
            all_ok, hard_ok = hits(labels, mixed, scores, valid)
            if best_linear is None or (hard_ok, all_ok) > (best_linear[0], best_linear[1]):
                best_linear = (hard_ok, all_ok, float(weight_rank), float(weight_pair), float(weight_mlp))
    print('best direct z', best_linear, flush=True)

    best_doc = None
    for alpha in grid:
        scores = alpha * zr + (1.0 - alpha) * zm
        all_ok, hard_ok = hits(labels, mixed, scores, valid)
        if best_doc is None or (hard_ok, all_ok) > (best_doc[0], best_doc[1]):
            best_doc = (hard_ok, all_ok, float(alpha))
    print('best lgbm+mlp only', best_doc, flush=True)

    best_temp = None
    temperatures = (0.25, 0.5, 1.0, 2.0, 4.0)
    coarse = np.linspace(0, 1, 11)
    for temperature in temperatures:
        pr, pp, pm = softmax(zr, temperature), softmax(zp, temperature), softmax(zm, temperature)
        for weight_rank in coarse:
            for weight_pair in coarse:
                weight_mlp = 1.0 - weight_rank - weight_pair
                if weight_mlp < -1e-9:
                    continue
                scores = weight_rank * pr + weight_pair * pp + weight_mlp * pm
                all_ok, hard_ok = hits(labels, mixed, scores, valid)
                if best_temp is None or (hard_ok, all_ok) > (best_temp[0], best_temp[1]):
                    best_temp = (
                        hard_ok, all_ok, float(temperature),
                        float(weight_rank), float(weight_pair), float(weight_mlp),
                    )
    print('best softmax temperature', best_temp, flush=True)

    raw_temps = (0.5, 1.0, 2.0, 4.0)
    best_raw = None
    for tr in raw_temps:
        for tp in raw_temps:
            for tm in raw_temps:
                pr, pp, pm = softmax(rank, tr), softmax(pair, tp), softmax(mlp, tm)
                for weight_rank in coarse:
                    for weight_pair in coarse:
                        weight_mlp = 1.0 - weight_rank - weight_pair
                        if weight_mlp < -1e-9:
                            continue
                        scores = weight_rank * pr + weight_pair * pp + weight_mlp * pm
                        all_ok, hard_ok = hits(labels, mixed, scores, valid)
                        if best_raw is None or (hard_ok, all_ok) > (best_raw[0], best_raw[1]):
                            best_raw = (
                                hard_ok, all_ok, tr, tp, tm,
                                float(weight_rank), float(weight_pair), float(weight_mlp),
                            )
    print('best raw-score temperatures', best_raw, flush=True)
    cleared = []
    for name, row in (
        ('direct_z', best_linear),
        ('lgbm_mlp', best_doc),
        ('softmax_z', best_temp),
        ('softmax_raw', best_raw),
    ):
        hard_ok, all_ok = row[0], row[1]
        ok = hard_ok >= BAR_HARD or (all_ok >= BAR_ALL and hard_ok >= CURRENT_HARD)
        cleared.append({'name': name, 'adopt': ok, 'hard_ok': hard_ok, 'all_ok': all_ok})
        print(name, 'ADOPT' if ok else 'keep', hard_ok, all_ok, flush=True)
    Path('runs/unused_search').mkdir(parents=True, exist_ok=True)
    Path('runs/unused_search/remaining.json').write_text(json.dumps({
        'bar_hard': BAR_HARD,
        'bar_all': BAR_ALL,
        'current_nested': hits(labels, mixed, current, valid),
        'direct_z': best_linear,
        'lgbm_mlp': best_doc,
        'softmax_z': best_temp,
        'softmax_raw': best_raw,
        'cleared': cleared,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
