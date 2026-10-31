"""Train the close-vote pairwise model.

Step and L2 are chosen on the seed-0 training holdout, by how well a mix
with the frozen blend does on disagreement questions. Validation is scored
after that choice. The mix replaces the submitted blend only when at least
two neighboring weights reach disagreement 302, or overall 745 with
disagreement still at least 298. Test labels are not read.
"""
import json
from pathlib import Path

import numpy as np

from close_model import (
    agreement,
    close_mask,
    close_scores,
    correct_count,
    frozen_blend_scores,
    logistic_fit,
    pair_arrays,
    standardize_fit,
)
from common import DATA, load_inputs, parse_answer, read_jsonl
from rank_features import _question_z, labels_from, mixed_mask, parsed_table
from train_lgbm import split_questions

LR = 0.2
MAX_STEPS = 80
L2_GRID = (1e-2, 1e-1, 1.0)
BAR_HARD = 302
BAR_ALL = 745
CURRENT_HARD = 298
BLEND = Path('runs/model_bundle_blend_mlp')
OUT = Path('runs/close_search')


def hits(scores, valid, labels, subset):
    return correct_count(scores, valid, labels, subset)


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
    close = close_mask(parsed)
    valid = np.array([[answer is not None for answer in row] for row in parsed])
    body, dev = split_questions(len(bank))
    body_close = body[close[body]]
    print(
        f'train={len(bank)} close={int(close.sum())} body_close={len(body_close)} '
        f'dev_mixed={int(mixed[dev].sum())} dev_close={int(close[dev].sum())}',
        flush=True,
    )
    print('scoring frozen blend on the training split', flush=True)
    base_scores, _ = frozen_blend_scores(bank, arrays['embeddings'], BLEND)
    dev_base_ok, dev_base_n = hits(base_scores[dev], valid[dev], labels[dev], mixed[dev])
    print(f'dev blend {dev_base_ok}/{dev_base_n}', flush=True)

    mean, std = standardize_fit(arrays['embeddings'], body_close)
    z_all = (arrays['embeddings'].astype(np.float64) - mean) / std
    vote, match = agreement(parsed)
    diffs, vote_gap, match_gap = pair_arrays(z_all, vote, match, labels, body_close)
    print(f'pairs={len(diffs)}', flush=True)
    dev_pack = (z_all[dev], vote[dev], match[dev], valid[dev], labels[dev], mixed[dev])

    # The frozen blend was fit on these development questions, so mixing
    # against it here favors weight 0. Choose the small model on its own.
    chosen = None
    rows = []
    for l2 in L2_GRID:
        print(f'=== l2 {l2}', flush=True)
        alone = logistic_fit(diffs, vote_gap, match_gap, dev_pack, MAX_STEPS, l2, LR)
        print(f'l2 {l2} alone {alone[0]}/{alone[5]} step {alone[1]}', flush=True)
        rows.append({'l2': l2, 'alone_ok': alone[0], 'alone_n': alone[5], 'step': alone[1]})
        candidate = (alone[0], -alone[1], float(l2))
        if chosen is None or candidate > chosen[0]:
            chosen = (candidate, alone[1], float(l2))

    print(f'chosen dev alone {rows} step {chosen[1]} l2 {chosen[2]}', flush=True)
    # Refit on every close training question, with fresh standardization.
    all_close = np.flatnonzero(close)
    mean, std = standardize_fit(arrays['embeddings'], all_close)
    z_all = (arrays['embeddings'].astype(np.float64) - mean) / std
    diffs, vote_gap, match_gap = pair_arrays(z_all, vote, match, labels, all_close)
    weight = np.zeros(diffs.shape[1])
    vote_coef = 0.0
    match_coef = 0.0
    n_pairs = len(diffs)
    for _ in range(chosen[1]):
        logit = diffs @ weight + vote_coef * vote_gap + match_coef * match_gap
        coeff = (1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))) - 1.0) / n_pairs
        weight -= LR * (diffs.T @ coeff + chosen[2] * weight)
        vote_coef -= LR * (float(vote_gap @ coeff) + chosen[2] * vote_coef)
        match_coef -= LR * (float(match_gap @ coeff) + chosen[2] * match_coef)
    params = {
        'weight': weight,
        'vote_coef': vote_coef,
        'match_coef': match_coef,
        'mean': mean,
        'std': std,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / 'close.npz',
        weight=weight.astype(np.float32),
        vote_coef=np.float32(vote_coef),
        match_coef=np.float32(match_coef),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    vbank, varrays = load_inputs('validation')
    vgold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    vparsed = parsed_table(vbank)
    vlabels = labels_from(vgold, vparsed)
    vmixed = mixed_mask(vparsed)
    vvalid = np.array([[answer is not None for answer in row] for row in vparsed])
    vbase, _ = frozen_blend_scores(vbank, varrays['embeddings'], BLEND)
    everyone = np.ones(len(vbank), dtype=bool)
    base_all, _ = hits(vbase, vvalid, vlabels, everyone)
    base_hard, base_hard_n = hits(vbase, vvalid, vlabels, vmixed)
    print(f'val frozen blend all {base_all} hard {base_hard}/{base_hard_n}', flush=True)
    vextra = close_scores(varrays['embeddings'], vparsed, params)
    alone_all, _ = hits(vextra, vvalid, vlabels, everyone)
    alone_hard, _ = hits(vextra, vvalid, vlabels, vmixed)
    print(f'val close model all {alone_all} hard {alone_hard}/{base_hard_n}', flush=True)
    sweep = []
    for alpha in np.linspace(0, 1, 21):
        mixed_scores = alpha * _question_z(vextra) + (1 - alpha) * _question_z(vbase)
        all_ok, _ = hits(mixed_scores, vvalid, vlabels, everyone)
        hard_ok, _ = hits(mixed_scores, vvalid, vlabels, vmixed)
        sweep.append({'alpha': float(alpha), 'all_ok': all_ok, 'hard_ok': hard_ok})
        print(f'alpha {alpha:.2f} all {all_ok} hard {hard_ok}', flush=True)
    meeting = [
        row for row in sweep
        if row['hard_ok'] >= BAR_HARD or (row['all_ok'] >= BAR_ALL and row['hard_ok'] >= CURRENT_HARD)
    ]
    alphas = [row['alpha'] for row in meeting]
    adjacent = any(abs(a - b) <= 0.051 and a != b for a in alphas for b in alphas)
    adopt = bool(meeting) and adjacent
    if meeting:
        meeting.sort(key=lambda row: (row['hard_ok'], row['all_ok'], -row['alpha']), reverse=True)
        picked = meeting[0]
    else:
        picked = max(sweep, key=lambda row: (row['hard_ok'], row['all_ok'], -row['alpha']))
    print(f'adopt {adopt} picked {picked}', flush=True)
    report = {
        'dev_blend': [dev_base_ok, dev_base_n],
        'grid': rows,
        'chosen_step': chosen[1],
        'chosen_l2': chosen[2],
        'vote_coef': vote_coef,
        'match_coef': match_coef,
        'val_alone': [alone_all, alone_hard],
        'val_blend': [base_all, base_hard],
        'sweep': sweep,
        'adopt': adopt,
        'picked': picked,
    }
    (OUT / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not adopt:
        print('keep current checkpoint', flush=True)
        return
    import shutil
    dest = Path('runs/model_bundle_close')
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('model.txt', 'preprocess.npz', 'linear.npz', 'mlp.npz'):
        shutil.copyfile(BLEND / name, dest / name)
    shutil.copyfile(OUT / 'close.npz', dest / 'close.npz')
    config = {
        'method': 'blend_close',
        'close_alpha': picked['alpha'],
        'mlp_alpha': 0.25,
        'rank_alpha': 0.45,
        'feature_mode': 'full',
        'selected_on': 'validation sweep after train-dev chose step and l2; neighboring weights cleared the bar',
    }
    (dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('wrote', dest, flush=True)


if __name__ == '__main__':
    main()
