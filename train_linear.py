"""Fit the embedding logistic regression on the training split only."""
import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_model import STEPS, LEARNING_RATE, L2, fit, save_bundle


def training_labels(bank, gold):
    labels = np.empty((len(bank), 4), dtype=np.float64)
    for i, (row, reference) in enumerate(zip(bank, gold)):
        gold_answer = parse_answer(reference['answer'])
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            labels[i, j] = float(answer is not None and answer == gold_answer)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=Path('runs/model_bundle'))
    args = parser.parse_args()
    bank, arrays = load_inputs('train', args.data_dir)
    gold = read_jsonl(Path(args.data_dir) / 'gsm8k_train.jsonl')
    if [row['id'] for row in gold] != [row['id'] for row in bank]:
        raise ValueError('Training references are not aligned with the bank')
    labels = training_labels(bank, gold)
    params = fit(arrays['embeddings'], labels)
    directory = save_bundle(args.output, params)
    positive = int(labels.sum())
    print(
        f'saved {directory} | steps={STEPS} lr={LEARNING_RATE} l2={L2} '
        f'positives={positive}/{labels.size}'
    )


if __name__ == '__main__':
    main()
