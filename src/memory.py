from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from src.env import Case
from typing import Dict, Tuple, Any

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

def bucket_K(k: int) -> str:
    if k <= 3:
        return "shallow"
    if k <=8:
        return "medium"
    return "deep"

Signature = Tuple[str, str, str, bool, str]
Key = Tuple[Signature, str]

def make_signature(case: Case) -> Signature:
    days_b = bucket_days(case.days_since_purchase)
    value_b = bucket_value(case.purchase_value)
    return (days_b, case.loyalty, case.product_type, case.damaged, value_b)

@dataclass
class EpisodicMemory:
    unseen_risk: float = 0.5
    seen: Dict[Key, int] = field(default_factory=lambda:defaultdict(int))
    failed: Dict[Key, int] = field(default_factory=lambda:defaultdict(int))

    def update(self, signature: Signature, *, K: int, correct: bool) -> None:
        key = (signature, bucket_K(K))
        self.seen[key] += 1
        if not correct:
            self.failed[key] += 1
        
    def risk(self, signature: Signature, K: int) -> float:
        key = (signature, bucket_K(K))
        n = self.seen[key]
        if n == 0:
            return self.unseen_risk
        return self.failed[key]/n
    
    def count(self, signature: Signature, K: int) -> int:
        key = (signature, bucket_K(K))
        return self.seen[key]



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
