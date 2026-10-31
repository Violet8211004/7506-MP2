"""Train a student on the frozen blend's soft scores.

The teacher is the submitted three-score blend. Soft targets use the training
split only. A student replaces that blend only if validation disagreement
reaches 302, or overall reaches 745 with disagreement still at least 298.
Test labels are not read.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from common import DATA, load_inputs, read_jsonl
from linear_model import load_bundle
from listwise_mlp import (
    fit_distill,
    load_mlp,
    predict_scores,
    refit_distill,
    save_mlp,
)
from rank_features import (
    _question_z,
    labels_from,
    load_stats,
    matrix_for,
    mixed_mask,
    parsed_table,
)
from submission import predict
from train_lgbm import split_questions

BAR_HARD = 302
BAR_ALL = 745
CURRENT_HARD = 298
CKPT = Path('runs/model_bundle_blend_mlp')


def teacher_distribution(bank, arrays, checkpoint):
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    features, valid, _ = matrix_for(
        bank, arrays['embeddings'], load_stats(checkpoint), config['feature_mode'],
    )
    rank = lgb.Booster(model_file=str(checkpoint / 'model.txt')).predict(features, num_threads=4)
    rank = np.asarray(rank, dtype=np.float64).reshape(len(bank), 4)
    linear = load_bundle(checkpoint)
    scale = np.where(linear['std'] == 0, 1.0, linear['std'])
    values = np.asarray(arrays['embeddings'], dtype=np.float64)
    pair = (values - linear['mean']) / scale @ linear['weight'] + linear['bias']
    params, stats = load_mlp(checkpoint)
    mlp = np.asarray(predict_scores(bank, arrays['embeddings'], params, stats), dtype=np.float64)
    blend = float(config['rank_alpha']) * _question_z(rank) + (1.0 - float(config['rank_alpha'])) * _question_z(pair)
    scores = float(config['mlp_alpha']) * _question_z(mlp) + (1.0 - float(config['mlp_alpha'])) * _question_z(blend)
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[~valid] = -1e9
    shift = scores.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(scores - shift, -50, 50))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def audit_shapes():
    bank, arrays = load_inputs('validation')
    for size in (1, 5, 17):
        choices = predict(
            bank[:size],
            {'embeddings': arrays['embeddings'][:size]},
            CKPT,
        )
        if len(choices) != size or any(type(choice) is not int or choice not in (0, 1, 2, 3) for choice in choices):
            raise RuntimeError(f'audit shape failed at n={size}: {choices}')
    print('audit shapes 1, 5, 17 ok', flush=True)


def score_student(params, stats):
    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    scores = np.asarray(predict_scores(bank, arrays['embeddings'], params, stats), dtype=np.float64)
    valid = np.array([[answer is not None for answer in row] for row in parsed])
    scores = scores.copy()
    scores[~valid] = -1e9
    choice = scores.argmax(axis=1)
    all_ok = int((labels[np.arange(len(choice)), choice] == 1).sum())
    idx = np.flatnonzero(mixed)
    hard_ok = int((labels[idx, choice[idx]] == 1).sum())
    return all_ok, hard_ok


def main():
    print(
        f'adopt student only if hard>={BAR_HARD} or (all>={BAR_ALL} and hard>={CURRENT_HARD})',
        flush=True,
    )
    audit_shapes()
    bank, arrays = load_inputs('train')
    gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    parsed = parsed_table(bank)
    labels = labels_from(gold, parsed)
    mixed = mixed_mask(parsed)
    body, dev = split_questions(len(bank))
    teacher = teacher_distribution(bank, arrays, CKPT)
    epochs, history = fit_distill(bank, arrays['embeddings'], teacher, labels, body, dev, mixed)
    print(f'refit epochs={epochs}', flush=True)
    params, stats = refit_distill(bank, arrays['embeddings'], teacher, mixed, epochs)
    directory = Path('runs/unused_search/distill_student')
    save_mlp(directory, params, stats, epochs, loss='distill')
    all_ok, hard_ok = score_student(params, stats)
    adopt = hard_ok >= BAR_HARD or (all_ok >= BAR_ALL and hard_ok >= CURRENT_HARD)
    print(f'VAL distill all {all_ok} hard {hard_ok} adopt {adopt}', flush=True)
    Path('runs/unused_search/distill.json').write_text(json.dumps({
        'epochs': epochs,
        'history': history,
        'all_ok': all_ok,
        'hard_ok': hard_ok,
        'adopt': adopt,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
