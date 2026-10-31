"""Small shared MLP with a listwise loss over the four candidates.

Training uses the training split only. Dropout is off at prediction.
"""
import json
from pathlib import Path

import numpy as np

from rank_features import hand_features, parsed_table


HIDDEN = 128
DROPOUT = 0.1
LEARNING_RATE = 1e-2
L2 = 1e-4
MAX_EPOCHS = 40
PATIENCE = 8
BATCH = 256
SEED = 42


def _standardize(values, mean, std):
    scale = np.where(std == 0, 1.0, std)
    return (values - mean) / scale


def _pack(embeddings, hand, emb_mean, emb_std, hand_mean, hand_std):
    left = _standardize(np.asarray(embeddings, dtype=np.float32), emb_mean, emb_std)
    right = _standardize(np.asarray(hand, dtype=np.float32), hand_mean, hand_std)
    return np.concatenate([left, right], axis=-1)


def _scores(x, params, rng=None):
    pre = x @ params['w1'] + params['b1']
    hidden = np.maximum(pre, 0.0)
    mask = None
    if rng is not None and DROPOUT > 0:
        mask = rng.random(hidden.shape) >= DROPOUT
        hidden = hidden * mask
    mean = hidden.mean(axis=1, keepdims=True)
    delta = hidden - mean
    dot = (hidden * mean).sum(axis=-1, keepdims=True)
    feat = np.concatenate([hidden, delta, dot], axis=-1)
    scores = feat @ params['w2'] + params['b2']
    cache = {'x': x, 'pre': pre, 'hidden': hidden, 'mean': mean, 'feat': feat, 'mask': mask}
    return scores, cache


def _listwise_grad(scores, labels):
    shift = scores.max(axis=1, keepdims=True)
    exp = np.exp(scores - shift)
    positive = exp * labels
    denom = positive.sum(axis=1, keepdims=True)
    keep = denom[:, 0] > 0
    prob = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)
    grad = prob - np.where(denom > 0, positive / np.clip(denom, 1e-8, None), 0.0)
    grad[~keep] = 0.0
    return grad.astype(np.float32), int(keep.sum())


def _backward(grad, cache, params):
    hidden = cache['hidden']
    mean = cache['mean']
    feat = cache['feat']
    dscore = grad[..., None]
    dw2 = feat.reshape(-1, feat.shape[-1]).T @ dscore.reshape(-1)
    db2 = dscore.sum()
    dfeat = dscore * params['w2']
    width = hidden.shape[-1]
    d_hidden = dfeat[..., :width]
    d_delta = dfeat[..., width:2 * width]
    d_dot = dfeat[..., -1]
    d_hidden = d_hidden + d_delta - d_delta.mean(axis=1, keepdims=True)
    d_hidden = d_hidden + d_dot[..., None] * mean
    d_hidden = d_hidden + (d_dot[..., None] * hidden).sum(axis=1, keepdims=True) / hidden.shape[1]
    if cache['mask'] is not None:
        d_hidden = d_hidden * cache['mask']
    d_hidden = d_hidden * (cache['pre'] > 0)
    x = cache['x']
    dw1 = x.reshape(-1, x.shape[-1]).T @ d_hidden.reshape(-1, width)
    db1 = d_hidden.sum(axis=(0, 1))
    return dw1, db1, dw2.reshape(-1), np.asarray(db2, dtype=np.float32).reshape(1)


class _Adam:
    def __init__(self, shapes, learning_rate):
        self.learning_rate = learning_rate
        self.step = 0
        self.m = [np.zeros(shape, dtype=np.float32) for shape in shapes]
        self.v = [np.zeros(shape, dtype=np.float32) for shape in shapes]

    def update(self, params, grads):
        self.step += 1
        out = {}
        for key, moment, second, grad in zip(params, self.m, self.v, grads):
            moment[:] = 0.9 * moment + 0.1 * grad
            second[:] = 0.999 * second + 0.001 * (grad * grad)
            moment_hat = moment / (1 - 0.9 ** self.step)
            second_hat = second / (1 - 0.999 ** self.step)
            out[key] = params[key] - self.learning_rate * moment_hat / (np.sqrt(second_hat) + 1e-8)
        return out


def _init(dim, rng):
    return {
        'w1': rng.normal(0, np.sqrt(2.0 / dim), size=(dim, HIDDEN)).astype(np.float32),
        'b1': np.zeros(HIDDEN, dtype=np.float32),
        'w2': rng.normal(0, 0.01, size=(HIDDEN * 2 + 1,)).astype(np.float32),
        'b2': np.zeros(1, dtype=np.float32),
    }


def _soft_grad(scores, target):
    shift = scores.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(scores - shift, -50, 50))
    prob = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)
    return (prob - target).astype(np.float32), scores.shape[0]


def _epoch(x, labels, params, adam, rng, indices, soft=False):
    rng.shuffle(indices)
    total = 0.0
    seen = 0
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start + BATCH]
        perm = np.argsort(rng.random((len(batch), 4)), axis=1)
        xb = x[batch]
        yb = labels[batch]
        xb = np.take_along_axis(xb, perm[..., None], axis=1)
        yb = np.take_along_axis(yb, perm, axis=1)
        scores, cache = _scores(xb, params, rng)
        grad, n_keep = _soft_grad(scores, yb) if soft else _listwise_grad(scores, yb)
        if n_keep == 0:
            continue
        dw1, db1, dw2, db2 = _backward(grad / max(n_keep, 1), cache, params)
        dw1 += L2 * params['w1']
        dw2 += L2 * params['w2']
        np.clip(dw1, -5, 5, out=dw1)
        np.clip(db1, -5, 5, out=db1)
        np.clip(dw2, -5, 5, out=dw2)
        params = adam.update(params, [dw1, db1, dw2, db2])
        total += float(n_keep)
        seen += len(batch)
    return params, total, seen


def disagreement_accuracy(bank, embeddings, labels, mixed, params, stats):
    scores = predict_scores(bank, embeddings, params, stats)
    valid = np.array([
        [answer is not None for answer in row]
        for row in parsed_table(bank)
    ])
    scores = scores.copy()
    scores[~valid] = -1e9
    choice = scores.argmax(axis=1)
    idx = np.flatnonzero(mixed)
    if len(idx) == 0:
        return 0.0, 0
    hit = int((labels[idx, choice[idx]] == 1).sum())
    return hit / len(idx), hit


def predict_scores(bank, embeddings, params, stats):
    hand = hand_features(bank, parsed_table(bank))
    x = _pack(embeddings, hand, stats['emb_mean'], stats['emb_std'], stats['hand_mean'], stats['hand_std'])
    scores, _ = _scores(x, params, None)
    return scores


def _valid_mask(bank):
    return np.array([
        [answer is not None for answer in row]
        for row in parsed_table(bank)
    ])


def predict_mlp(bank, arrays, checkpoint):
    checkpoint = Path(checkpoint)
    params, stats = load_mlp(checkpoint)
    scores = predict_scores(bank, arrays['embeddings'], params, stats)
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[~_valid_mask(bank)] = -np.inf
    return [int(i) for i in np.argmax(scores, axis=1)]


def predict_blend_mlp(bank, arrays, checkpoint):
    """z(MLP) mixed with the already z-scored ranker-plus-linear blend."""
    import lightgbm as lgb

    from linear_model import load_bundle
    from rank_features import _question_z, load_stats, matrix_for

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    mlp_alpha = float(config['mlp_alpha'])
    rank_alpha = float(config['rank_alpha'])
    params, mlp_stats = load_mlp(checkpoint)
    mlp_scores = predict_scores(bank, arrays['embeddings'], params, mlp_stats)

    features, valid, _ = matrix_for(
        bank, arrays['embeddings'], load_stats(checkpoint), config['feature_mode'],
    )
    rank_scores = lgb.Booster(model_file=str(checkpoint / 'model.txt')).predict(
        features, num_threads=4,
    )
    rank_scores = np.asarray(rank_scores, dtype=np.float64).reshape(len(bank), 4)
    linear = load_bundle(checkpoint)
    scale = np.where(linear['std'] == 0, 1.0, linear['std'])
    values = np.asarray(arrays['embeddings'], dtype=np.float64)
    if linear.get('center'):
        values = values - values.mean(axis=1, keepdims=True)
    pair_scores = (values - linear['mean']) / scale @ linear['weight'] + linear['bias']
    blend = rank_alpha * _question_z(rank_scores) + (1.0 - rank_alpha) * _question_z(pair_scores)
    scores = mlp_alpha * _question_z(mlp_scores) + (1.0 - mlp_alpha) * _question_z(blend)
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[~valid] = -np.inf
    return [int(i) for i in np.argmax(scores, axis=1)]


def fit_mlp(bank, embeddings, labels, body, dev, mixed):
    """Early-stop on the development disagreement slice, then return that step count."""
    hand = hand_features(bank, parsed_table(bank))
    body_rows = hand[body].reshape(-1, hand.shape[-1])
    emb_rows = np.asarray(embeddings[body], dtype=np.float32).reshape(-1, embeddings.shape[-1])
    stats = {
        'emb_mean': emb_rows.mean(axis=0),
        'emb_std': emb_rows.std(axis=0),
        'hand_mean': body_rows.mean(axis=0),
        'hand_std': body_rows.std(axis=0),
    }
    x = _pack(embeddings, hand, stats['emb_mean'], stats['emb_std'], stats['hand_mean'], stats['hand_std'])
    rng = np.random.default_rng(SEED)
    params = _init(x.shape[-1], rng)
    adam = _Adam([params[key].shape for key in ('w1', 'b1', 'w2', 'b2')], LEARNING_RATE)
    train_idx = np.flatnonzero(mixed[body])
    train_questions = body[train_idx]
    best = None
    best_params = {key: value.copy() for key, value in params.items()}
    wait = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        params, _, _ = _epoch(x, labels, params, adam, rng, train_questions.copy())
        dev_score, dev_hit = disagreement_accuracy(
            [bank[i] for i in dev], embeddings[dev], labels[dev], mixed[dev], params, stats,
        )
        history.append({'epoch': epoch, 'dev_hard': dev_hit, 'dev_n': int(mixed[dev].sum()), 'dev_acc': dev_score})
        print(f'mlp epoch {epoch} dev_hard {dev_hit}/{int(mixed[dev].sum())}', flush=True)
        if best is None or dev_hit > best:
            best = dev_hit
            best_params = {key: value.copy() for key, value in params.items()}
            wait = 0
            kept = epoch
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_params, stats, kept, history


def _train_stats(hand, embeddings, rows):
    return {
        'emb_mean': np.asarray(embeddings[rows], dtype=np.float32).reshape(-1, embeddings.shape[-1]).mean(axis=0),
        'emb_std': np.asarray(embeddings[rows], dtype=np.float32).reshape(-1, embeddings.shape[-1]).std(axis=0),
        'hand_mean': hand[rows].reshape(-1, hand.shape[-1]).mean(axis=0),
        'hand_std': hand[rows].reshape(-1, hand.shape[-1]).std(axis=0),
    }


def fit_distill(bank, embeddings, teacher, gold, body, dev, mixed):
    """Match the teacher distribution. The kept epoch uses development gold accuracy."""
    hand = hand_features(bank, parsed_table(bank))
    stats = _train_stats(hand, embeddings, body)
    x = _pack(embeddings, hand, stats['emb_mean'], stats['emb_std'], stats['hand_mean'], stats['hand_std'])
    rng = np.random.default_rng(SEED)
    params = _init(x.shape[-1], rng)
    adam = _Adam([params[key].shape for key in ('w1', 'b1', 'w2', 'b2')], LEARNING_RATE)
    train_questions = body[np.flatnonzero(mixed[body])]
    best = None
    wait = 0
    kept = 1
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        params, _, _ = _epoch(x, teacher, params, adam, rng, train_questions.copy(), soft=True)
        _, dev_hit = disagreement_accuracy(
            [bank[i] for i in dev], embeddings[dev], gold[dev], mixed[dev], params, stats,
        )
        history.append({'epoch': epoch, 'dev_hard': dev_hit, 'dev_n': int(mixed[dev].sum())})
        print(f'distill epoch {epoch} dev_hard {dev_hit}/{int(mixed[dev].sum())}', flush=True)
        if best is None or dev_hit > best:
            best = dev_hit
            wait = 0
            kept = epoch
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return kept, history


def refit_distill(bank, embeddings, teacher, mixed, epochs):
    hand = hand_features(bank, parsed_table(bank))
    stats = _train_stats(hand, embeddings, np.arange(len(bank)))
    x = _pack(embeddings, hand, stats['emb_mean'], stats['emb_std'], stats['hand_mean'], stats['hand_std'])
    rng = np.random.default_rng(SEED)
    params = _init(x.shape[-1], rng)
    adam = _Adam([params[key].shape for key in ('w1', 'b1', 'w2', 'b2')], LEARNING_RATE)
    train_questions = np.flatnonzero(mixed)
    for epoch in range(1, epochs + 1):
        params, _, _ = _epoch(x, teacher, params, adam, rng, train_questions.copy(), soft=True)
        print(f'distill refit {epoch}/{epochs}', flush=True)
    return params, stats


def refit_mlp(bank, embeddings, labels, mixed, epochs):
    hand = hand_features(bank, parsed_table(bank))
    rows = hand.reshape(-1, hand.shape[-1])
    emb_rows = np.asarray(embeddings, dtype=np.float32).reshape(-1, embeddings.shape[-1])
    stats = {
        'emb_mean': emb_rows.mean(axis=0),
        'emb_std': emb_rows.std(axis=0),
        'hand_mean': rows.mean(axis=0),
        'hand_std': rows.std(axis=0),
    }
    x = _pack(embeddings, hand, stats['emb_mean'], stats['emb_std'], stats['hand_mean'], stats['hand_std'])
    rng = np.random.default_rng(SEED)
    params = _init(x.shape[-1], rng)
    adam = _Adam([params[key].shape for key in ('w1', 'b1', 'w2', 'b2')], LEARNING_RATE)
    train_questions = np.flatnonzero(mixed)
    for epoch in range(1, epochs + 1):
        params, _, _ = _epoch(x, labels, params, adam, rng, train_questions.copy())
        print(f'mlp refit {epoch}/{epochs}', flush=True)
    return params, stats


def save_mlp(directory, params, stats, epochs, loss='listwise'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / 'mlp.npz', **params, **stats)
    config = {
        'method': 'mlp',
        'epochs': int(epochs),
        'hidden': HIDDEN,
        'dropout': DROPOUT,
        'learning_rate': LEARNING_RATE,
        'l2': L2,
        'seed': SEED,
        'loss': loss,
    }
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')


def load_mlp(checkpoint):
    path = Path(checkpoint)
    with np.load(path / 'mlp.npz') as bundle:
        params = {key: bundle[key] for key in ('w1', 'b1', 'w2', 'b2')}
        stats = {key: bundle[key] for key in ('emb_mean', 'emb_std', 'hand_mean', 'hand_std')}
    return params, stats
