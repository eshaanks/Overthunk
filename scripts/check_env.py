from random import Random
from collections import Counter
from src.env import sample_case, ground_truth

rng = Random(0)
counts = Counter()
for _ in range(10000):
    c = sample_case(rng)
    counts[ground_truth(c)] += 1
print(counts)