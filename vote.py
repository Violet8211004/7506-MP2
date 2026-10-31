"""Majority vote over parsed final numbers. No learned asset."""
from common import parse_answer


def predict(bank, arrays, checkpoint):
    """Pick the smallest index whose final number has the highest count."""
    del arrays, checkpoint
    choices = []
    for row in bank:
        parsed = [parse_answer(candidate['text']) for candidate in row['candidates']]
        counts = {}
        for answer in parsed:
            if answer is not None:
                counts[answer] = counts.get(answer, 0) + 1
        if not counts:
            choices.append(0)
            continue
        best = max(counts.values())
        choices.append(next(
            i for i, answer in enumerate(parsed)
            if answer is not None and counts[answer] == best
        ))
    return choices
