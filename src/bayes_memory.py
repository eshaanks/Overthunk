from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

from src.env import Case

Signature = Tuple[str, str, str, bool, str]


def bucket_days(days_since_purchase: int) -> str:
    if days_since_purchase <= 14:
        return "0-14"
    if days_since_purchase <= 30:
        return "15-30"
    if days_since_purchase <= 45:
        return "31-45"
    return "46-60"


def bucket_value(purchase_value: float) -> str:
    if purchase_value <= 100:
        return "10-100"
    if purchase_value <= 500:
        return "101-500"
    return "501-1000"


def make_signature(case: Case) -> Signature:
    return (
        bucket_days(case.days_since_purchase),
        case.loyalty,
        case.product_type,
        case.damaged,
        bucket_value(case.purchase_value),
    )


def _sigmoid(z: float) -> float:
    # numerically stable-ish for our small z values
    return 1.0 / (1.0 + math.exp(-z))

@dataclass
class BayesianSigmoidMemory:
     """
    Per-signature Bayesian logistic regression with diagonal Gaussian posterior
    over theta = (alpha, beta).

    Model: P(fail | sig, K) = sigmoid(alpha - beta * log(K))
    """
     prior_var = 4.0

     mu_a: Dict[Signature, float] = field(default_factory=dict)
     mu_b: Dict[Signature, float] = field(default_factory=dict)
     prec_a: Dict[Signature, float] = field(default_factory=dict)
     prec_b: Dict[Signature, float] = field(default_factory=dict)

     def _ensure(self, sig: Signature) -> None:
         if sig in self.mu_a:
             return
         
         self.mu_a[sig] = 0.0
         self.mu_b[sig] = 1.0  # start with "compute helps"
         p0 = 1.0 / self.prior_var
         self.prec_a[sig] = p0
         self.prec_b[sig] = p0

     def risk_mean(self, sig: Signature, K: int) -> float:
         self._ensure(sig)
         a = self.mu_a
         b = self.mu_b
         z = a - b * math.log(K)
         return _sigmoid(z)
     
     def sample_theta(self, sig: Signature, rng) -> tuple[float, float]:
         self._ensure(sig)
         var_a = 1/self.prec_a[sig]
         var_b = 1/self.prec_b[sig]
         a = rng.gauss(self.mu_a[sig], math.sqrt(var_a))
         b = rng.gauss(self.mu_b[sig], math.sqrt(var_b))
         # enforce monotonicity direction: more compute shouldn't increase risk
         b = max(b, 0.05)
         return a, b
     
     def update(self, sig: Signature, *, K: int, correct: bool) -> None:
        """
        One-step diagonal Laplace/ADF-style update.

        Let y=1 if fail else 0.
        x = [1, -logK]
        p = sigmoid(mu·x)
        w = p(1-p)
        precision += w * x^2
        mean += (y - p) * x / precision   (elementwise)
        """
        self._ensure(sig)
        y = 0 if correct else 1
        x1 = 1.0
        x2 = -math.log(K)
        a = self.mu_a[sig]
        b = self.mu_b[sig]
        z = a + b * x2
        p = _sigmoid(z)
        w = p * (1 - p)
         # update precisions (diag Hessian)
        self.prec_a[sig] += w * (x1 * x1)
        self.prec_b[sig] += w * (x2 * x2)

        # update means (diag Newton step / assumed density filtering)
        err = (y - p)
        self.mu_a[sig] += (err * x1) / self.prec_a[sig]
        self.mu_b[sig] += (err * x2) / self.prec_b[sig]

        # keep beta from collapsing negative
        self.mu_b[sig] = max(self.mu_b[sig], 0.05)
          