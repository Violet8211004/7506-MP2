"""Pointwise L2 logistic regression on the supplied 1024-d embeddings.

The fit is deterministic: full-batch gradient descent on the entire training
split, weights and bias initialised at zero. Validation and test labels are
never read here.
"""
import json
from pathlib import Path

import numpy as np

from common import parse_answer

STEPS = 80
LEARNING_RATE = 0.2
L2 = 1e-2
BUNDLE_NAME = 'linear.npz'
CONFIG_NAME = 'config.json'


def fit(embeddings, labels):
    """Fit on embeddings of shape [N, 4, 1024] and labels of shape [N, 4]."""
    x = np.asarray(embeddings, dtype=np.float64).reshape(-1, 1024)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (x - mean) / std
    n, dim = z.shape
    weight = np.zeros(dim, dtype=np.float64)
    bias = np.float64(0.0)
    for _ in range(STEPS):
        probability = 1.0 / (1.0 + np.exp(-np.clip(z @ weight + bias, -50, 50)))
        error = (probability - y) / n
        grad_w = z.T @ error + L2 * weight
        weight -= LEARNING_RATE * grad_w
        bias -= LEARNING_RATE * error.sum()
    return {
        'weight': weight.astype(np.float32),
        'bias': np.float32(bias),
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
    }


def save_bundle(directory, params):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / BUNDLE_NAME,
        weight=params['weight'],
        bias=np.asarray(params['bias'], dtype=np.float32),
        mean=params['mean'],
        std=params['std'],
    )
    config = {
        'method': 'pointwise_l2_logistic',
        'steps': STEPS,
        'learning_rate': LEARNING_RATE,
        'l2': L2,
        'l2_on': 'weight only; loss term is (l2/2) * ||weight||^2',
        'init': 'zeros',
        'standardize': 'training-set mean and population standard deviation',
        'unparsed': 'trained with label 0; excluded from the argmax at inference',
        'tie_break': 'smallest candidate index',
    }
    (directory / CONFIG_NAME).write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return directory


def load_bundle(checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        path = path / BUNDLE_NAME
    with np.load(path) as bundle:
        center = bool(bundle['center']) if 'center' in bundle.files else False
        return {
            'weight': bundle['weight'].astype(np.float64),
            'bias': float(bundle['bias']),
            'mean': bundle['mean'].astype(np.float64),
            'std': bundle['std'].astype(np.float64),
            'center': center,
        }


def parsed_mask(bank):
    mask = np.empty((len(bank), 4), dtype=bool)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            mask[i, j] = parse_answer(candidate['text']) is not None
    return mask


def choose(bank, embeddings, params):
    """Score each candidate and return one Python int in 0..3 per question."""
    std = np.where(params['std'] == 0, 1.0, params['std'])
    values = np.asarray(embeddings, dtype=np.float64)
    if params.get('center'):
        values = values - values.mean(axis=1, keepdims=True)
    centered = (values - params['mean']) / std
    scores = centered @ params['weight'] + params['bias']
    scores = scores.copy()
    scores[~parsed_mask(bank)] = -np.inf
    return [int(i) for i in np.argmax(scores, axis=1)]
