from collections import Counter


def get_class_stats(targets):
    counter = Counter()
    for target in targets:
        counter += Counter(target)
    counts = dict(counter)
    total = counter.total()
    frequencies = {k: v / total if total != 0 else 0 for k, v in counts.items()}
    weights = {k: 1 / v if v != 0 else 0 for k, v in frequencies.items()}
    return counts, frequencies, weights
