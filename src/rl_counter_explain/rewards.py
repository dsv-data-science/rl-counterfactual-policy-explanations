"""
Shaped reward functions.
"""

import copy
from typing import Any, Callable

from tf_agents.trajectories import time_step as ts

from rl_counter_explain import record

RewardShapeFn = Callable[[record.BlackboxCounterfactual, record.Step, Any], ts.TimeStep]


def create_identity_reward_fn():
    """
    Returns the reward, unmodified.
    """

    def shape_reward_fn(
        tutor_student: record.BlackboxCounterfactual,
        step: record.Step,
        policy_state: Any,
    ) -> ts.TimeStep:
        """
        Returns:
            The step.next_time_step without making alterations.
        """
        del tutor_student
        del policy_state
        return copy.deepcopy(step.next_time_step)

    return shape_reward_fn


def create_time_step_shaped_reward_fn(reward_shape_fn: Callable[[ts.TimeStep], float]):
    """
    Applies a reward factor based on the `next_time_step`.
    """

    def shape_reward_fn(
        blackbox_counterfactual: record.BlackboxCounterfactual,
        step: record.Step,
        policy_state: Any,
    ):
        del blackbox_counterfactual
        del policy_state
        return step.next_time_step._replace(reward=reward_shape_fn(step.next_time_step))

    return shape_reward_fn


def apply_intervention_reward(reward_fn: RewardShapeFn, penalty: float):
    """
    Applies a penalty to the reward when the blackbox and counterfactual actions
    differ.

    Note: penalty is deducted, hence, it should be a non-negative value.
    """

    assert penalty >= 0, "penalty should be a non-negative value."

    def reward_intervention_fn(
        blackbox_counterfactual: record.BlackboxCounterfactual,
        step: record.Step,
        policy_state: Any,
    ):
        next_time_step = reward_fn(blackbox_counterfactual, step, policy_state)
        student_step = blackbox_counterfactual.counterfactual_policy.action(
            step.time_step, policy_state
        )
        tutor_step = blackbox_counterfactual.blackbox_policy.action(
            step.time_step, policy_state
        )
        if student_step.action != tutor_step.action:
            return next_time_step._replace(reward=step.next_time_step.reward - penalty)
        return copy.deepcopy(next_time_step)

    return reward_intervention_fn
