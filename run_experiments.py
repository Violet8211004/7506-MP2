"""Formal comparisons E0–E9.

Validation labels are read only after every model and the E8 coefficients
are frozen. The test split is never opened.
"""
import argparse
import hashlib
import re
import time
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from linear_model import L2, LEARNING_RATE, STEPS, fit as fit_e1, parsed_mask
from vote import predict as vote_predict

HASH_DIM = 1 << 16
HASH_SEED = 0
GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MLP_STEPS = 80
MLP_LR = 0.05
MLP_L2 = 1e-4
MLP_CLIP = 5.0
ROOT = Path(__file__).resolve().parents[1]

NUM = r'\d[\d,]*(?:\.\d+)?'
OP = r'(?:\+|−|-|\\times|\\div|\*|×|÷|/)'
EXPR = re.compile(rf'({NUM}(?:\s*{OP}\s*{NUM})+)\s*=\s*([-+]?{NUM})')
FRAC = re.compile(rf'\\frac\{{([^{{}}]+)\}}\{{([^{{}}]+)\}}\s*=\s*([-+]?{NUM})')
TOKEN = re.compile(r'[A-Za-z]+|\d+(?:\.\d+)?')
AFFIRM = re.compile(
    r'\b(?:therefore|thus|clearly|definitely)\b|final answer|the answer is',
    re.IGNORECASE,
)
CHECKS = re.compile('[✓✔✅√]')


def labels_for(bank, gold):
    labels = np.empty((len(bank), 4), dtype=np.float64)
    for i, (row, reference) in enumerate(zip(bank, gold)):
        gold_answer = parse_answer(reference['answer'])
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            labels[i, j] = float(answer is not None and answer == gold_answer)
    return labels


def parsed_answers(bank):
    return [
        [parse_answer(candidate['text']) for candidate in row['candidates']]
        for row in bank
    ]


def vote_count_and_margin(answers):
    counts = {}
    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
    per_candidate = [0 if answer is None else counts[answer] for answer in answers]
    if not counts:
        return per_candidate, 0
    ordered = sorted(counts.values(), reverse=True)
    second = ordered[1] if len(ordered) > 1 else 0
    return per_candidate, ordered[0] - second


def split_questions(n, seed=0):
    order = np.random.default_rng(seed).permutation(n)
    n_dev = n // 10
    dev = order[:n_dev]
    body = order[n_dev:]
    return body, dev


def choices_from_scores(bank, scores):
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~parsed_mask(bank)] = -np.inf
    return [int(i) for i in np.argmax(masked, axis=1)]


def linear_scores(features, params):
    std = np.where(params['std'] == 0, 1.0, params['std'])
    centered = (np.asarray(features, dtype=np.float64) - params['mean']) / std
    return centered @ params['weight'] + float(params['bias'])


def fit_linear(features, labels):
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
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
        weight -= LEARNING_RATE * (z.T @ error + L2 * weight)
        bias -= LEARNING_RATE * error.sum()
    return {
        'weight': weight.astype(np.float32),
        'bias': np.float32(bias),
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
    }


def bundle_bytes(params, with_mean=True):
    total = np.asarray(params['weight'], dtype=np.float32).nbytes
    total += np.asarray(params['bias'], dtype=np.float32).nbytes
    if with_mean:
        total += np.asarray(params['mean'], dtype=np.float32).nbytes
        total += np.asarray(params['std'], dtype=np.float32).nbytes
    return int(total)


def metrics(bank, gold, choices):
    hard_ok = hard_n = m0_ok = m0_n = m1_ok = m1_n = low_ok = low_n = correct = 0
    for row, reference, choice in zip(bank, gold, choices):
        gold_answer = parse_answer(reference['answer'])
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        hit = [answer is not None and answer == gold_answer for answer in answers]
        _, margin = vote_count_and_margin(answers)
        chosen = hit[choice]
        correct += int(chosen)
        if sum(hit) < 4:
            hard_n += 1
            hard_ok += int(chosen)
        if margin == 0:
            m0_n += 1
            m0_ok += int(chosen)
        elif margin == 1:
            m1_n += 1
            m1_ok += int(chosen)
        if margin <= 1:
            low_n += 1
            low_ok += int(chosen)
    return {
        'correct': correct,
        'n': len(bank),
        'hard_ok': hard_ok,
        'hard_n': hard_n,
        'm0_ok': m0_ok,
        'm0_n': m0_n,
        'm1_ok': m1_ok,
        'm1_n': m1_n,
        'low_ok': low_ok,
        'low_n': low_n,
    }


def pp(ok, n):
    return 100.0 * ok / n


def slice_bank(bank, idx):
    return [bank[i] for i in idx]


def close(left, right):
    return abs(left - right) <= 1e-6 * max(1.0, abs(right))


def _as_float(text):
    try:
        return float(str(text).replace(',', '').replace('−', '-'))
    except ValueError:
        return None


def eval_lhs(lhs):
    text = (lhs.replace('−', '-').replace('\\times', '*').replace('\\div', '/')
            .replace('×', '*').replace('÷', '/').replace(',', ''))
    parts = re.findall(r'\d+(?:\.\d+)?|[+\-*/]', text)
    if len(parts) < 3:
        return None
    numbers = []
    ops = []
    expect_number = True
    for part in parts:
        if expect_number:
            if part in '+-*/':
                return None
            numbers.append(float(part))
            expect_number = False
        else:
            if part not in '+-*/':
                return None
            ops.append(part)
            expect_number = True
    if expect_number:
        return None
    values = [numbers[0]]
    add_ops = []
    for op, right in zip(ops, numbers[1:]):
        if op in '*/':
            if op == '/' and right == 0:
                return None
            values[-1] = values[-1] * right if op == '*' else values[-1] / right
        else:
            add_ops.append(op)
            values.append(right)
    total = values[0]
    for op, right in zip(add_ops, values[1:]):
        total = total + right if op == '+' else total - right
    return total


def equation_checks(text):
    good = bad = 0
    for lhs, rhs in EXPR.findall(text):
        predicted = eval_lhs(lhs)
        actual = _as_float(rhs)
        if predicted is None or actual is None:
            continue
        if close(predicted, actual):
            good += 1
        else:
            bad += 1
    for left, right, rhs in FRAC.findall(text):
        numerator = _as_float(left)
        denominator = _as_float(right)
        actual = _as_float(rhs)
        if None in (numerator, denominator, actual) or denominator == 0:
            continue
        if close(numerator / denominator, actual):
            good += 1
        else:
            bad += 1
    return good, bad


def consistency_fraction(text):
    good, bad = equation_checks(text)
    total = good + bad
    return 0.0 if total == 0 else good / total


def answer_span(text):
    text = str(text)
    if '####' in text:
        return text.rsplit('####', 1)[1]
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    return boxed[-1] if boxed else ''


def reasoning_text(text):
    text = str(text)
    if '####' in text:
        return text.rsplit('####', 1)[0]
    boxed = list(re.finditer(r'\\boxed\{[^{}]+\}', text))
    if not boxed:
        return text
    last = boxed[-1]
    return text[:last.start()] + text[last.end():]


def norm_numbers(text):
    found = []
    for raw in re.findall(NUM, text):
        value = _as_float(raw)
        if value is not None:
            found.append(round(value, 6))
    return found


def coverage(question, text):
    wanted = set(norm_numbers(question))
    if not wanted:
        return 0.0
    seen = set(norm_numbers(reasoning_text(text)))
    return len(wanted & seen) / len(wanted)


def surface_features(bank):
    features = np.empty((len(bank), 4, 5), dtype=np.float64)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            text = candidate['text']
            features[i, j] = (
                np.log1p(len(text)),
                text.count('**'),
                text.count('\\boxed'),
                len(CHECKS.findall(text)),
                len(AFFIRM.findall(text)),
            )
    return features


def vote_features(bank):
    features = np.empty((len(bank), 4, 2), dtype=np.float64)
    for i, row in enumerate(bank):
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts, _ = vote_count_and_margin(answers)
        for j, answer in enumerate(answers):
            features[i, j] = (counts[j], float(answer is not None))
    return features


def consistency_features(bank):
    features = np.empty((len(bank), 4, 3), dtype=np.float64)
    for i, row in enumerate(bank):
        answers = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts, _ = vote_count_and_margin(answers)
        for j, candidate in enumerate(row['candidates']):
            text = candidate['text']
            features[i, j] = (
                consistency_fraction(text),
                coverage(row['question'], text),
                counts[j],
            )
    return features


def hash_index(token):
    digest = hashlib.md5(bytes([HASH_SEED]) + token.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little') % HASH_DIM


def hash_rows(texts):
    rows = []
    for text in texts:
        seen = set()
        indexes = []
        for token in TOKEN.findall(text.lower()):
            index = hash_index(token)
            if index not in seen:
                seen.add(index)
                indexes.append(index)
        rows.append(np.asarray(indexes, dtype=np.int32))
    return rows


def texts_of(bank, answer_only):
    texts = []
    for row in bank:
        for candidate in row['candidates']:
            texts.append(answer_span(candidate['text']) if answer_only else candidate['text'])
    return texts


def hashed_scores(rows, weight, bias):
    scores = np.full(len(rows), float(bias), dtype=np.float64)
    for i, indexes in enumerate(rows):
        if len(indexes):
            scores[i] += weight[indexes].sum()
    return scores


def fit_hashed(rows, labels):
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    n = len(rows)
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int32, count=n)
    nonempty = [row for row in rows if len(row)]
    flat = np.concatenate(nonempty) if nonempty else np.zeros(0, dtype=np.int32)
    owners = np.repeat(np.arange(n), lengths)
    weight = np.zeros(HASH_DIM, dtype=np.float64)
    bias = np.float64(0.0)
    for _ in range(STEPS):
        logits = np.full(n, bias, dtype=np.float64)
        if flat.size:
            np.add.at(logits, owners, weight[flat])
        error = (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))) - y) / n
        grad = np.zeros(HASH_DIM, dtype=np.float64)
        if flat.size:
            np.add.at(grad, flat, np.repeat(error, lengths))
        weight -= LEARNING_RATE * (grad + L2 * weight)
        bias -= LEARNING_RATE * float(error.sum())
    return weight.astype(np.float32), np.float32(bias)


def standardize(values, mean, std):
    std = np.where(std == 0, 1.0, std)
    return (values - mean) / std


def fit_pairwise(z_body, y_body, z_dev, dev_bank, dev_labels):
    diffs = []
    for zi, yi in zip(z_body, y_body):
        positive = np.flatnonzero(yi == 1)
        negative = np.flatnonzero(yi == 0)
        for a in positive:
            for b in negative:
                diffs.append(zi[a] - zi[b])
    diffs = np.asarray(diffs, dtype=np.float64)
    n_pairs, dim = diffs.shape
    weight = np.zeros(dim, dtype=np.float64)
    best_correct = -1
    best = weight.copy()
    best_step = 0
    for step in range(STEPS + 1):
        scores = z_dev @ weight
        correct = metrics_correct(dev_bank, dev_labels, choices_from_scores(dev_bank, scores))
        if correct > best_correct:
            best_correct = correct
            best = weight.copy()
            best_step = step
        if step == STEPS:
            break
        margin = diffs @ weight
        coeff = (1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50))) - 1.0) / n_pairs
        weight -= LEARNING_RATE * (diffs.T @ coeff + L2 * weight)
    return {
        'weight': best.astype(np.float32),
        'bias': np.float32(0.0),
        'step': best_step,
    }


def mlp_forward(z, params):
    flat = np.asarray(z, dtype=np.float64).reshape(-1, z.shape[-1])
    hidden = flat @ params['W1'] + params['b1']
    hidden = np.maximum(hidden, 0.0)
    logits = hidden @ params['W2'] + params['b2']
    return logits.reshape(z.shape[:-1])


def fit_mlp(z_body, y_body, z_dev, dev_bank, dev_labels):
    rng = np.random.default_rng(0)
    dim = z_body.shape[-1]
    hidden = 64
    params = {
        'W1': rng.normal(0.0, np.sqrt(2.0 / dim), size=(dim, hidden)),
        'b1': np.zeros(hidden),
        'W2': rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden,)),
        'b2': np.float64(0.0),
    }
    flat = z_body.reshape(-1, dim)
    y = y_body.reshape(-1)
    n = flat.shape[0]
    best_correct = -1
    best = None
    best_step = 0
    for step in range(MLP_STEPS + 1):
        scores = mlp_forward(z_dev, params)
        correct = metrics_correct(dev_bank, dev_labels, choices_from_scores(dev_bank, scores))
        if correct > best_correct:
            best_correct = correct
            best = {key: np.array(value, copy=True) for key, value in params.items()}
            best_step = step
        if step == MLP_STEPS:
            break
        pre = flat @ params['W1'] + params['b1']
        active = pre > 0
        hid = pre * active
        logits = hid @ params['W2'] + params['b2']
        error = (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))) - y) / n
        grad_w2 = hid.T @ error + MLP_L2 * params['W2']
        grad_b2 = error.sum()
        grad_h = error[:, None] * params['W2']
        grad_h *= active
        grad_w1 = flat.T @ grad_h + MLP_L2 * params['W1']
        grad_b1 = grad_h.sum(axis=0)
        norm = np.sqrt(
            grad_w1.ravel() @ grad_w1.ravel()
            + grad_b1 @ grad_b1
            + grad_w2 @ grad_w2
            + grad_b2 ** 2
        )
        if norm > MLP_CLIP:
            scale = MLP_CLIP / norm
            grad_w1 *= scale
            grad_b1 *= scale
            grad_w2 *= scale
            grad_b2 *= scale
        params['W1'] -= MLP_LR * grad_w1
        params['b1'] -= MLP_LR * grad_b1
        params['W2'] -= MLP_LR * grad_w2
        params['b2'] -= MLP_LR * grad_b2
    best['step'] = best_step
    return best


def metrics_correct(bank, labels, choices):
    return sum(int(labels[i][choice] == 1) for i, choice in enumerate(choices))


def std_of(x):
    flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_std(x, mean, std):
    return (np.asarray(x, dtype=np.float64) - mean) / std


def self_check():
    assert consistency_fraction('10+15+5=30') == 1
    assert consistency_fraction('15+5=30') == 0
    assert consistency_fraction(r'5 \times 17 = 85') == 1
    assert consistency_fraction(r'\frac{7}{1.4} = 5') == 1
    assert consistency_fraction('no equations here') == 0
    assert close(eval_lhs('0-5'), -5)


def holdout_gate(dev, low, val, e1):
    dev_ok = dev >= e1['dev'] + 0.5 or low >= e1['low'] + 2.0
    val_ok = val >= e1['val']
    return dev_ok and val_ok


def preregistered_gate(val, low, e1):
    if val >= e1['val'] + 0.5:
        return True
    return low >= e1['low'] + 2.0 and val >= e1['val']


def fmt(ok, n):
    return f'{pp(ok, n):.2f}%（{ok}/{n}）'


def write_outputs(rows, e1_gate, notes):
    rows = sorted(rows, key=lambda row: row['method'])
    out = Path('runs')
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'experiments.csv'
    header = (
        'method,val_all,val_hard,val_margin_0,val_margin_1,'
        'asset_bytes,train_seconds,predict_seconds'
    )
    lines = [header]
    for row in rows:
        result = row['val']
        lines.append(','.join([
            row['method'],
            f'{pp(result["correct"], result["n"]):.2f}',
            f'{pp(result["hard_ok"], result["hard_n"]):.2f}',
            f'{pp(result["m0_ok"], result["m0_n"]):.2f}',
            f'{pp(result["m1_ok"], result["m1_n"]):.2f}',
            str(row['asset_bytes']),
            f'{row["train_seconds"]:.3f}',
            f'{row["predict_seconds"]:.3f}',
        ]))
    csv_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    winners = [row for row in rows if row['gate']]
    winners.sort(key=lambda row: (
        -row['val']['correct'],
        -row['val']['hard_ok'],
        row['asset_bytes'],
    ))
    if winners:
        winner = winners[0]
        decision = (
            f"按事先写好的规则，可以替换向量打分的是 **{winner['method']} {winner['name']}**。"
            f"验证集整体 {fmt(winner['val']['correct'], winner['val']['n'])}，"
            f"向量打分是 {fmt(e1_gate['correct'], e1_gate['n'])}。"
        )
        if len(winners) > 1:
            others = '、'.join(f"{row['method']}" for row in winners[1:])
            decision += f" 同样跨过门槛、但验证集整体更低的是 {others}。"
    else:
        decision = (
            '按事先写好的规则，没有方法可以替换向量打分。提交维持 **E1 向量打分**。'
        )
    body = [
        '# 正式实验：对照结果',
        '',
        '验证集每个方法只评了一次。测试集没有打开。准确率来自和官方评分器相同的规则：选中解答的最终数字与参考答案一致即算对。',
        '',
        '替换规则在看验证集之前就定好了。',
        '',
        '- E7 成对比较、E8 投票加打分、E9 小网络用训练集里留出的 10% 做早停或选系数。要替换 E1，开发集整体至少高 0.5 个百分点，或者开发集里领先票数不超过 1 的题至少高 2 个百分点，并且验证集整体不低于 E1。',
        '- E2、E3、E4、E5、E6 的配方事先写死，在全部训练集上拟合。要替换 E1，验证集整体至少高 0.5 个百分点，或者领先票数不超过 1 的题至少高 2 个百分点且验证集整体不低于 E1。',
        '- 几个方法都满足时，取得验证集整体最高的那个。都不满足，就交 E1。',
        '',
        '开发集由种子 0 的题目排列切出，4755 题中的前 475 题，模型拟合时不用这 475 题的标签做早停以外的事。E1、E2、E3、E4、E5、E6 为了和已复现的向量打分可比，仍用全部训练集拟合，所以它们的开发集数字不是留出成绩，不拿来当门槛。',
        '',
        '| 方法 | 验证集整体 | 难题子集 | 票数打平 | 只领先 1 票 | 资产字节 | 训练秒 | 预测秒 | 替换 E1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in rows:
        result = row['val']
        body.append('| ' + ' | '.join([
            f"{row['method']} {row['name']}",
            fmt(result['correct'], result['n']),
            fmt(result['hard_ok'], result['hard_n']),
            fmt(result['m0_ok'], result['m0_n']),
            fmt(result['m1_ok'], result['m1_n']),
            str(row['asset_bytes']),
            f"{row['train_seconds']:.3f}",
            f"{row['predict_seconds']:.3f}",
            row['gate_text'],
        ]) + ' |')
    body.extend(['', '## 判断', '', decision, '', '## 每个方法在看什么', ''])
    for row in rows:
        body.append(f"- **{row['method']} {row['name']}。** {notes[row['method']]} {row.get('detail', '')}")
    body.extend([
        '',
        '资产字节是推理时必须保存的 float32 参数和标准化统计量，不含 npz 文件头。E0 没有参数。预测秒是验证集上构造特征和打分的时间，不含读入数据文件。',
        '',
        '机器可读的同一张表在 `code/runs/experiments.csv`。',
        '',
    ])
    path = ROOT / '正式实验-对照结果.md'
    path.write_text('\n'.join(body), encoding='utf-8')
    print(f'wrote {csv_path}')
    print(f'wrote {path}')
    print(decision)


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()
    self_check()
    train_bank, train_arrays = load_inputs('train')
    train_gold = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if [row['id'] for row in train_gold] != [row['id'] for row in train_bank]:
        raise ValueError('Training references are not aligned with the bank')
    train_labels = labels_for(train_bank, train_gold)
    body_idx, dev_idx = split_questions(len(train_bank))
    dev_bank = slice_bank(train_bank, dev_idx)
    dev_labels = train_labels[dev_idx]
    print(f'split body={len(body_idx)} dev={len(dev_idx)}', flush=True)

    e1_params, e1_train = timed(lambda: fit_e1(train_arrays['embeddings'], train_labels))
    e1_dev_scores = linear_scores(train_arrays['embeddings'][dev_idx], e1_params)
    e1_dev_choices = choices_from_scores(dev_bank, e1_dev_scores)
    e1_dev = metrics(dev_bank, slice_bank(train_gold, dev_idx), e1_dev_choices)
    print(
        f"E1 dev {e1_dev['correct']}/{e1_dev['n']} "
        f"low {e1_dev['low_ok']}/{e1_dev['low_n']} train_s={e1_train:.3f}",
        flush=True,
    )

    print('features', flush=True)
    surface, surface_s = timed(lambda: surface_features(train_bank))
    votes, votes_s = timed(lambda: vote_features(train_bank))
    consist, consist_s = timed(lambda: consistency_features(train_bank))
    full_rows, full_s = timed(lambda: hash_rows(texts_of(train_bank, False)))
    span_rows, span_s = timed(lambda: hash_rows(texts_of(train_bank, True)))
    print(
        f'feature seconds surface={surface_s:.1f} vote={votes_s:.1f} '
        f'consist={consist_s:.1f} full={full_s:.1f} span={span_s:.1f}',
        flush=True,
    )

    linear_jobs = {}
    for name, features, prep in (
        ('E2', surface, surface_s),
        ('E3', votes, votes_s),
        ('E6', consist, consist_s),
    ):
        params, seconds = timed(lambda features=features: fit_linear(features, train_labels))
        linear_jobs[name] = (params, features, seconds + prep)
        dev_scores = linear_scores(features[dev_idx], params)
        dev_correct = metrics_correct(dev_bank, dev_labels, choices_from_scores(dev_bank, dev_scores))
        print(f'{name} dev correct {dev_correct}/{len(dev_bank)}', flush=True)

    print('hash fit', flush=True)
    full_fit, full_fit_s = timed(lambda: fit_hashed(full_rows, train_labels))
    span_fit, span_fit_s = timed(lambda: fit_hashed(span_rows, train_labels))
    print(f'hash fit seconds full={full_fit_s:.1f} span={span_fit_s:.1f}', flush=True)

    body_emb = train_arrays['embeddings'][body_idx]
    mean, std = std_of(body_emb)
    z_body = apply_std(body_emb, mean, std)
    z_dev = apply_std(train_arrays['embeddings'][dev_idx], mean, std)
    print('pairwise', flush=True)
    pairwise, pair_s = timed(lambda: fit_pairwise(
        z_body, train_labels[body_idx], z_dev, dev_bank, dev_labels,
    ))
    print(f"E7 step={pairwise['step']} seconds={pair_s:.1f}", flush=True)
    print('mlp', flush=True)
    mlp, mlp_s = timed(lambda: fit_mlp(
        z_body, train_labels[body_idx], z_dev, dev_bank, dev_labels,
    ))
    print(f"E9 step={mlp['step']} seconds={mlp_s:.1f}", flush=True)

    e1_all = linear_scores(train_arrays['embeddings'], e1_params)
    vote_only = votes[:, :, 0]
    fuse_mean = np.stack([
        e1_all[body_idx].reshape(-1).mean(),
        vote_only[body_idx].reshape(-1).mean(),
    ])
    fuse_std = np.stack([
        e1_all[body_idx].reshape(-1).std(),
        vote_only[body_idx].reshape(-1).std(),
    ])
    fuse_std = np.where(fuse_std == 0, 1.0, fuse_std)
    z_dev_e1 = (e1_all[dev_idx] - fuse_mean[0]) / fuse_std[0]
    z_dev_vote = (vote_only[dev_idx] - fuse_mean[1]) / fuse_std[1]
    pure = choices_from_scores(dev_bank, z_dev_e1)
    if pure != e1_dev_choices:
        raise RuntimeError('Standardized E1 scores changed the dev choices')
    best_key = None
    best_coef = (1.0, 0.0)
    fuse_start = time.perf_counter()
    for c1 in GRID:
        for c2 in GRID:
            choices = choices_from_scores(dev_bank, c1 * z_dev_e1 + c2 * z_dev_vote)
            correct = metrics_correct(dev_bank, dev_labels, choices)
            low_bank = []
            low_labels = []
            low_choices = []
            for local, question in enumerate(dev_idx):
                answers = [
                    parse_answer(candidate['text'])
                    for candidate in train_bank[question]['candidates']
                ]
                _, margin = vote_count_and_margin(answers)
                if margin <= 1:
                    low_bank.append(dev_bank[local])
                    low_labels.append(dev_labels[local])
                    low_choices.append(choices[local])
            low_correct = metrics_correct(low_bank, low_labels, low_choices) if low_bank else 0
            key = (correct, low_correct, c1, -c2)
            if best_key is None or key > best_key:
                best_key = key
                best_coef = (c1, c2)
    fuse_seconds = time.perf_counter() - fuse_start
    print(f'E8 coef={best_coef} dev_correct={best_key[0]} low={best_key[1]}', flush=True)
    dev_gold = slice_bank(train_gold, dev_idx)
    c1, c2 = best_coef
    holdout_dev = {
        'E7': metrics(dev_bank, dev_gold, choices_from_scores(
            dev_bank, z_dev @ pairwise['weight'].astype(np.float64),
        )),
        'E8': metrics(dev_bank, dev_gold, choices_from_scores(
            dev_bank, c1 * z_dev_e1 + c2 * z_dev_vote,
        )),
        'E9': metrics(dev_bank, dev_gold, choices_from_scores(
            dev_bank, mlp_forward(z_dev, mlp),
        )),
    }
    for name, dev_result in holdout_dev.items():
        print(
            f"{name} frozen dev {dev_result['correct']}/{dev_result['n']} "
            f"low {dev_result['low_ok']}/{dev_result['low_n']}",
            flush=True,
        )

    if args.skip_validation:
        print('skip validation')
        return

    val_bank, val_arrays = load_inputs('validation')
    val_gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    if [row['id'] for row in val_gold] != [row['id'] for row in val_bank]:
        raise ValueError('Validation references are not aligned with the bank')

    prepared = {
        'full_rows': full_rows,
        'span_rows': span_rows,
        'full_fit': full_fit,
        'span_fit': span_fit,
        'full_s': full_s + full_fit_s,
        'span_s': span_s + span_fit_s,
        'mean': mean,
        'std': std,
        'pairwise': pairwise,
        'pair_s': pair_s,
        'mlp': mlp,
        'mlp_s': mlp_s,
        'coef': best_coef,
        'fuse_mean': fuse_mean,
        'fuse_std': fuse_std,
        'fuse_s': fuse_seconds,
        'e1_params': e1_params,
        'e1_train': e1_train,
        'linear': linear_jobs,
        'holdout_dev': holdout_dev,
    }
    rows = score_validation(val_bank, val_gold, val_arrays, prepared, e1_dev)
    by_name = {row['method']: row for row in rows}
    if by_name['E1']['val']['correct'] != 719 or by_name['E0']['val']['correct'] != 688:
        raise RuntimeError('E0/E1 did not reproduce 688 and 719; table was not written')
    notes = {
        'E0': '只数每份解答的最终数字，选出现次数最多的那份，次数相同就选下标最小的。',
        'E1': '用课程给的 1024 维向量做线性打分。80 步、学习率 0.2、L2 系数 0.01，权重从 0 开始，在全部训练集上拟合。',
        'E2': '只看字数、加粗、有没有方框、对勾和肯定用语，线性配方与 E1 相同。',
        'E3': '只看这个最终数字出现了几次，以及能不能解析出数字。线性配方与 E1 相同。',
        'E4': '把整段原文变成 65536 维的词哈希，再用与 E1 相同的步数、学习率和 L2 拟合。不另存均值和标准差，所以资产约 256 KiB。',
        'E5': '哈希方式和 E4 相同，输入只保留 `####` 后面或最后一个方框里的那段文字。',
        'E6': '检查算式是否算对、题面里的数字有没有在推理里出现，再加上票数。线性配方与 E1 相同。',
        'E7': f"在训练主体上比较同一题里正确解答和错误解答的向量差，开发集上选早停步数，选中的是第 {pairwise['step']} 步。",
        'E8': f"分数 = {best_coef[0]:g} × 标准化后的 E1 分数 + {best_coef[1]:g} × 标准化后的票数。系数只在开发集的网格上选择。",
        'E9': f"1024 维向量经过一层宽度 64 的网络。开发集上选早停步数，选中的是第 {mlp['step']} 步。学习率 0.05，L2 系数 1e-4，种子 0。",
    }
    write_outputs(rows, by_name['E1']['val'], notes)


def score_validation(val_bank, val_gold, val_arrays, prepared, e1_dev):
    e1_gate = {
        'dev': pp(e1_dev['correct'], e1_dev['n']),
        'low': pp(e1_dev['low_ok'], e1_dev['low_n']),
    }

    def add_predict(name, name_zh, train_seconds, asset, predict_fn, gate_kind):
        choices, seconds = timed(predict_fn)
        result = metrics(val_bank, val_gold, choices)
        if gate_kind == 'ref':
            gate = False
            gate_text = '基准' if name == 'E0' else '当前要交的'
        elif gate_kind == 'holdout':
            dev_result = prepared['holdout_dev'][name]
            gate = holdout_gate(
                pp(dev_result['correct'], dev_result['n']),
                pp(dev_result['low_ok'], dev_result['low_n']),
                pp(result['correct'], result['n']),
                e1_gate,
            )
            gate_text = '是' if gate else '否'
        else:
            low = pp(result['low_ok'], result['low_n'])
            gate = preregistered_gate(pp(result['correct'], result['n']), low, {
                'val': pp(rows[1]['val']['correct'], rows[1]['val']['n']),
                'low': pp(rows[1]['val']['low_ok'], rows[1]['val']['low_n']),
            })
            gate_text = '是' if gate else '否'
        rows.append({
            'method': name,
            'name': name_zh,
            'val': result,
            'asset_bytes': asset,
            'train_seconds': train_seconds,
            'predict_seconds': seconds,
            'gate': gate,
            'gate_text': gate_text,
            'choices': choices,
        })
        print(
            f"{name} val {result['correct']}/{result['n']} "
            f"hard {result['hard_ok']}/{result['hard_n']}",
            flush=True,
        )

    rows = []
    add_predict('E0', '投票', 0.0, 0, lambda: vote_predict(val_bank, None, None), 'ref')
    add_predict(
        'E1', '向量打分', prepared['e1_train'], bundle_bytes(prepared['e1_params']),
        lambda: choices_from_scores(
            val_bank, linear_scores(val_arrays['embeddings'], prepared['e1_params']),
        ),
        'ref',
    )
    e1_gate['val'] = pp(rows[1]['val']['correct'], rows[1]['val']['n'])
    e1_gate['low'] = pp(rows[1]['val']['low_ok'], rows[1]['val']['low_n'])
    e1_gate['correct'] = rows[1]['val']['correct']
    e1_gate['n'] = rows[1]['val']['n']

    for name, name_zh in (('E2', '表面特征'), ('E3', '票数'), ('E6', '等式与覆盖')):
        params, features_unused, train_seconds = prepared['linear'][name]
        del features_unused

        def predict(params=params, name=name):
            if name == 'E2':
                features = surface_features(val_bank)
            elif name == 'E3':
                features = vote_features(val_bank)
            else:
                features = consistency_features(val_bank)
            return choices_from_scores(val_bank, linear_scores(features, params))

        add_predict(name, name_zh, train_seconds, bundle_bytes(params), predict, 'pre')

    def predict_hash(answer_only, fit):
        rows_h = hash_rows(texts_of(val_bank, answer_only))
        weight, bias = fit
        scores = hashed_scores(rows_h, weight.astype(np.float64), bias).reshape(len(val_bank), 4)
        return choices_from_scores(val_bank, scores)

    weight, bias = prepared['full_fit']
    add_predict(
        'E4', '全文哈希', prepared['full_s'], int(weight.nbytes + np.asarray(bias).nbytes),
        lambda: predict_hash(False, prepared['full_fit']), 'pre',
    )
    weight, bias = prepared['span_fit']
    add_predict(
        'E5', '只留最终答案', prepared['span_s'], int(weight.nbytes + np.asarray(bias).nbytes),
        lambda: predict_hash(True, prepared['span_fit']), 'pre',
    )

    def predict_pair():
        z = apply_std(val_arrays['embeddings'], prepared['mean'], prepared['std'])
        scores = z @ prepared['pairwise']['weight'].astype(np.float64)
        return choices_from_scores(val_bank, scores)

    pair_asset = int(
        prepared['pairwise']['weight'].nbytes
        + prepared['mean'].astype(np.float32).nbytes
        + prepared['std'].astype(np.float32).nbytes
    )
    add_predict('E7', '成对比较', prepared['pair_s'], pair_asset, predict_pair, 'holdout')

    def predict_fuse():
        e1_scores = linear_scores(val_arrays['embeddings'], prepared['e1_params'])
        vote_counts = vote_features(val_bank)[:, :, 0]
        z_e1 = (e1_scores - prepared['fuse_mean'][0]) / prepared['fuse_std'][0]
        z_vote = (vote_counts - prepared['fuse_mean'][1]) / prepared['fuse_std'][1]
        c1, c2 = prepared['coef']
        return choices_from_scores(val_bank, c1 * z_e1 + c2 * z_vote)

    fuse_asset = bundle_bytes(prepared['e1_params']) + 6 * 4
    add_predict('E8', '投票加打分', prepared['fuse_s'], fuse_asset, predict_fuse, 'holdout')

    def predict_mlp():
        z = apply_std(val_arrays['embeddings'], prepared['mean'], prepared['std'])
        return choices_from_scores(val_bank, mlp_forward(z, prepared['mlp']))

    mlp = prepared['mlp']
    mlp_asset = int(
        mlp['W1'].astype(np.float32).nbytes + mlp['b1'].astype(np.float32).nbytes
        + mlp['W2'].astype(np.float32).nbytes + np.float32(mlp['b2']).nbytes
        + prepared['mean'].astype(np.float32).nbytes
        + prepared['std'].astype(np.float32).nbytes
    )
    add_predict('E9', '小网络', prepared['mlp_s'], mlp_asset, predict_mlp, 'holdout')
    return rows


if __name__ == '__main__':
    main()
