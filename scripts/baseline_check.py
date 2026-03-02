from random import Random
from src.env import sample_case, ground_truth

def baseline(case):
    # base heuristic: days<=30 approve else deny
    return "approve" if case.days_since_purchase <= 30 else "deny"

rng = Random(0)
correct = 0
N = 10000
for _ in range(N):
    c = sample_case(rng)
    y = ground_truth(c)
    a = baseline(c)
    correct += (a == y)

print("baseline_acc:", correct / N)