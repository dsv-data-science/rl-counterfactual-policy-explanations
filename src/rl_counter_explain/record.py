"""
Modeule contains structures for counterfactual policy evaluation.
"""

import dataclasses

from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts


@dataclasses.dataclass(frozen=True)
class BlackboxCounterfactual:
    """
    Class has a tutor and student policies.
    """

    blackbox_policy: py_policy.PyPolicy
    counterfactual_policy: py_policy.PyPolicy


@dataclasses.dataclass(frozen=True)
class Step:
    """
    Class tracks time_steps before and after a policy step.
    """

    time_step: ts.TimeStep
    policy_step: policy_step.PolicyStep
    next_time_step: ts.TimeStep
