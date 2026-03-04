from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from src.env import Case
from typing import Dict, Tuple, Any
import math

def bucket_days(days_since_purchase: int) -> str:
    if days_since_purchase <= 14:
        return "0-14"
    if days_since_purchase <= 30:
        return "15-30"
    if days_since_purchase <= 45:
        return "16-45"
    return "46-60"

def bucket_value(purchase_value: float) -> str:
    if purchase_value <= 100:
        return "10-100"
    if purchase_value <= 500:
        return "101-500"
    return "501-100"

Signature = Tuple[str, str, str, bool, str]

def make_signature(case: Case) -> Signature:
    days_b = bucket_days(case.days_since_purchase)
    value_b = bucket_value(case.purchase_value)
    return (days_b, case.loyalty, case.product_type, case.damaged, value_b)

@dataclass
class EpisodicMemory:
    lr: float = 0.05  # learning rate

    alpha: Dict[Signature, float] = field(default_factory=dict)
    beta: Dict[Signature, float] = field(default_factory=dict)

    def _ensure_params(self, signature: Signature):
        if signature not in self.alpha:
            self.alpha[signature] = 0.0
            self.beta[signature] = 1.0

    def update(self, signature: Signature, *, K: int, correct: bool) -> None:
        self._ensure_params(signature)

        # convert to failure label
        y = 0 if correct else 1

        a = self.alpha[signature]
        b = self.beta[signature]

        # forward pass
        z = a - b * math.log(K)
        r_hat = 1.0 / (1.0 + math.exp(-z))

        # gradient term
        error = r_hat - y

        # gradient step
        self.alpha[signature] -= self.lr * error
        self.beta[signature] -= self.lr * error * (-math.log(K))
        
    def risk(self, signature: Signature, K: int) -> float:
        self._ensure_params(signature)

        a = self.alpha[signature]
        b = self.beta[signature]

        z = a - b * math.log(K)
        return 1.0 / (1.0 + math.exp(-z))



if __name__ == "__main__":
    from random import Random
    from src.env import sample_case, ground_truth

    rng = Random(0)
    mem = EpisodicMemory()

    case = sample_case(rng)
    sig = make_signature(case)
    gt = ground_truth(case)

    print("case:", case)
    print("signature:", sig)
    print("ground truth:", gt)

    # Before any updates, risk should be unseen_risk (0.5)
    for K in (1, 5, 12):
        print(f"initial risk K={K} ->", mem.risk(sig, K), "count:", mem.count(sig, K))

    # Simulate a failure at deep compute
    mem.update(sig, K=12, correct=False)
    print("after deep fail: risk K=12 ->", mem.risk(sig, 12), "count:", mem.count(sig, 12))

    # Simulate a success at deep compute
    mem.update(sig, K=12, correct=True)
    print("after deep fail+success: risk K=12 ->", mem.risk(sig, 12), "count:", mem.count(sig, 12))

    # Simulate two shallow outcomes
    mem.update(sig, K=1, correct=False)
    mem.update(sig, K=1, correct=True)
    print("after shallow fail+success: risk K=1 ->", mem.risk(sig, 1), "count:", mem.count(sig, 1))
