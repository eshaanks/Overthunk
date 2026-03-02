# src/env.py
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

Loyalty = Literal["low", "medium", "high"]
ProductType = Literal["electronics", "clothing", "consumable"]
Action = Literal["approve", "deny", "escalate"]

ACTIONS: tuple[Action, ...] = ("approve", "deny", "escalate")
LOYALTIES: tuple[Loyalty, ...] = ("low", "medium", "high")
PRODUCT_TYPES: tuple[ProductType, ...] = ("electronics", "clothing", "consumable")


@dataclass(frozen=True)
class Case:
    days_since_purchase: int         # 0..60
    loyalty: Loyalty                 # low/medium/high
    product_type: ProductType        # electronics/clothing/consumable
    damaged: bool                    # True/False
    purchase_value: int              # 10..1000


def sample_case(rng: Random) -> Case:
    """Generate one random case (repeatable if rng is seeded)."""
    return Case(
        days_since_purchase=rng.randint(0, 60),
        loyalty=rng.choice(LOYALTIES),
        product_type=rng.choice(PRODUCT_TYPES),
        damaged=rng.choice([True, False]),
        purchase_value=rng.randint(10, 1000),
    )


def ground_truth(case: Case) -> Action:
    """
    Deterministic label function with explicit precedence (top overrides bottom).

    Rules (in order):
    1) If damaged -> approve
    2) If consumable and days > 14 -> deny
    3) If value > 500 and loyalty high -> escalate
    4) If loyalty high and days <= 45 -> escalate
    5) If days <= 30 -> approve
    6) Else -> deny
    """
    days = case.days_since_purchase

    # 1) Damaged always approved (highest precedence)
    if case.damaged:
        return "approve"

    # 2) Consumables after 14 days always denied
    if case.product_type == "consumable" and days > 14:
        return "deny"

    # 3) High-value + high-loyalty escalates
    if case.purchase_value > 500 and case.loyalty == "high":
        return "escalate"

    # 4) High-loyalty exception up to 45 days escalates
    if case.loyalty == "high" and days <= 45:
        return "escalate"

    # 5) Base policy
    if days <= 30:
        return "approve"

    # 6) Default
    return "deny"


def case_to_str(case: Case) -> str:
    return (
        f"days={case.days_since_purchase}, loyalty={case.loyalty}, "
        f"type={case.product_type}, damaged={case.damaged}, "
        f"value={case.purchase_value}"
    )


if __name__ == "__main__":
    rng = Random(0)
    for _ in range(5):
        c = sample_case(rng)
        print(case_to_str(c), "=>", ground_truth(c))