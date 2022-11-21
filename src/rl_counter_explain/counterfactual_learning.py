"""
Using Q-learning to learn a counterfactual policy.

Problem: RedGreeSeq
"""

import argparse
import collections
import copy
import dataclasses
import json
import logging
import os
import os.path
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing.types import NestedArray, Seed

from rlplg import tracking
from rlplg.environments.gridworld import constants as constants_gridworld
from rlplg.environments.gridworld import env as env_gridworld
from rlplg.environments.gridworld import utils as utils_gridworld
from rlplg.environments.redgreen import constants as constants_redgreen
from rlplg.environments.redgreen import env as env_redgreen
from rlplg.examples import qlearning
from rlplg.learning.tabular import dynamicprog, markovdp, policies
from rl_counter_explain import envstats
from rl_counter_explain import record, rewards

REDGREEN = "redgreen"
GRIDWORLD = "gridworld"
RANDOM = "random"
ALWAYS_GREEN = "always-green"
DYNA_PROG = "dyna-prog"
QLEARNING = "q-learning"
SARSA = "sarsa"
SAME_FN = "same-fn"
COUNTERFACTUAL_REWARD_FN = "counterfactual-reward-fn"
ON = "on"
OFF = "off"
MDP_STATS_EPISODES = 1_000_000


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Program arguments

    Args:
        num_episodes - the num of episodes for counterfactual policy control (policy optimization).
    """

    num_episodes: int
    num_stats_episodes: int
    problem: str
    mdp_stats_path: str
    output_path: str
    reward_fn: str
    control_fn: str
    intervention_penalty: str
    counterfactual_trajectory_prob: float


class CounterfactualQGreedyPolicy(policies.PyPolicy):
    """
    A counterfactual policy - can intervene on a blackbox policy's actions to achieve a specific objective.
    """

    def __init__(
        self,
        blackbox_policy: policies.PyPolicy,
        state_id_fn: Callable[[NestedArray, NestedArray], int],
        action_values: np.ndarray,
    ):
        super().__init__(
            time_step_spec=blackbox_policy.time_step_spec,
            action_spec=blackbox_policy.action_spec,
            emit_log_probability=False,
        )
        self._blackbox_policy = blackbox_policy
        self._state_id_fn = state_id_fn
        self._state_action_value_table = copy.deepcopy(action_values)

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: NestedArray = (),
        seed: Optional[Seed] = None,
    ) -> policy_step.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")

        blackbox_policy_step = self._blackbox_policy.action(
            time_step, policy_state=policy_state
        )
        state_id = self._state_id_fn(time_step.observation, blackbox_policy_step.action)
        action = np.argmax(self._state_action_value_table[state_id])
        policy_info = ()

        return policy_step.PolicyStep(
            action=action.astype(self.action_spec.dtype),
            state=policy_state,
            info=policy_info,
        )


class BlackboxCounterfactualStats:
    """
    A class to accumulate statistics on the intervention
    of a counterfactual policy upon a blackbox policy.
    """

    def __init__(
        self,
        blackbox_policy: policies.PyPolicy,
        counterfactual_policy: policies.PyPolicy,
        state_id_fn: Callable[[np.ndarray], int],
    ):
        self._blackbox_policy = blackbox_policy
        self._counterfactual_policy = counterfactual_policy
        self._state_id_fn = state_id_fn

        self._num_events = 0
        self._state_visits = collections.defaultdict(int)
        self._blackbox_policy_state_action_count = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self._counterfactual_policy_state_action_count = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self._counterfactual_policy_state_intervention_count = collections.defaultdict(
            int
        )

    def accumulate(
        self, time_step: ts.TimeStep, policy_state: NestedArray = ()
    ) -> policy_step.PolicyStep:
        """
        Accumulates stats on the interventions of the counterfactual policy on the actions
        of the blackbox policy.
        """
        state_id = self._state_id_fn(time_step.observation)
        blackbox_policy_step = self._blackbox_policy.action(time_step, policy_state)
        counterfactual_policy_step = self._counterfactual_policy.action(
            time_step, policy_state
        )

        self._num_events += 1
        self._state_visits[state_id] += 1
        self._blackbox_policy_state_action_count[state_id][
            blackbox_policy_step.action
        ] += 1
        self._counterfactual_policy_state_action_count[state_id][
            counterfactual_policy_step.action
        ] += 1
        if blackbox_policy_step.action != counterfactual_policy_step.action:
            self._counterfactual_policy_state_intervention_count[state_id] += 1

    def state_stats(self) -> Mapping[int, "StateStats"]:
        """
        Returns intervention stats for each state.
        """
        interventions = {}
        for state_id, state_visits in self._state_visits.items():
            blackbox_policy_action_stats = {
                action_id: count / state_visits
                for action_id, count in self._blackbox_policy_state_action_count[
                    state_id
                ].items()
            }
            counterfactual_and_blackbox_policy_actions = set(
                list(self._blackbox_policy_state_action_count[state_id].keys())
                + list(self._counterfactual_policy_state_action_count[state_id].keys())
            )
            counterfactual_policy_action_stats = {
                action_id: self._counterfactual_policy_state_action_count[state_id].get(
                    action_id, 0
                )
                / state_visits
                for action_id in counterfactual_and_blackbox_policy_actions
            }
            counterfactual_policy_state_intervention_prob = (
                self._counterfactual_policy_state_intervention_count.get(state_id, 0)
                / state_visits
            )
            interventions[state_id] = StateStats(
                blackbox_policy_action_stats=blackbox_policy_action_stats,
                counterfactual_policy_action_stats=counterfactual_policy_action_stats,
                counterfactual_policy_intervention_prob=counterfactual_policy_state_intervention_prob,
            )
        return interventions

    @property
    def num_counterfactual_interventions(self) -> int:
        """
        Returns the total number of times the counterfactual policy
        intervened on a blackbox policy's action.
        """
        return sum(self._counterfactual_policy_state_intervention_count.values())


@dataclasses.dataclass(frozen=True)
class StateStats:
    """
    Stats for a specific state on:
      - Actions of the blackbox policy.
      - Actions of the counterfactual policy.
      - Counterfactual policy interventions.
    """

    blackbox_policy_action_stats: Mapping[int, float]
    counterfactual_policy_action_stats: Mapping[int, float]
    counterfactual_policy_intervention_prob: float

    def __str__(self):
        return json.dumps(self.__dict__)


class ProblemSpec:
    """
    Class holds configuration for a problem, e.g. environment, MDP.
    """

    def __init__(
        self,
        environment: py_environment.PyEnvironment,
        mdp: markovdp.MDP,
        blackbox_policy: py_policy.PyPolicy,
        state_id_fn: Callable[[NestedArray], int],
        success_fn: Callable[[ts.TimeStep], bool],
        reward_shaping_functions: Mapping[str, rewards.RewardShapeFn],
    ):
        for fn_type in (SAME_FN, COUNTERFACTUAL_REWARD_FN):
            assert (
                fn_type in reward_shaping_functions
            ), f"reward_shaping_functions is missing fn_type: {fn_type}"
        self._environment = environment
        self._mdp = mdp
        self._blackbox_policy = blackbox_policy
        # TODO: use env spec instead of env? or pass discretizer or is sthis something else?
        self._state_id_fn = state_id_fn
        self._success_fn = success_fn
        self._reward_shaping_fns = reward_shaping_functions

    @property
    def environment(self) -> py_environment.PyEnvironment:
        """
        Data collection env.
        """
        return self._environment

    @property
    def mdp(self) -> markovdp.MDP:
        """
        Implementation of the MDP.
        """
        return self._mdp

    @property
    def blackbox_policy(self) -> py_policy.PyPolicy:
        """
        A blackbox policy.
        """
        return self._blackbox_policy

    def state_id_fn(self, observation: NestedArray) -> int:
        """
        Maps observations to an integer state ID.
        """
        return self._state_id_fn(observation)

    def success_fn(self, time_step: ts.TimeStep) -> bool:
        """
        True if the episode ended successfully.
        """
        return self._success_fn(time_step)

    def reward_shaping_fn(self, fn_type: str) -> rewards.RewardShapeFn:
        """
        Returns a reward shaping function.

        Raises:
            KeyError is the requested function is undefined.
        """
        return self._reward_shaping_fns[fn_type]


def parse_args() -> Args:
    """
    Parses program arguments.

    Returns:
        An instance of `Args`.
    """
    arg_parser = argparse.ArgumentParser(prog="Counterfactual Policy Example")
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument(
        "--problem", type=str, default=GRIDWORLD, choices=[REDGREEN, GRIDWORLD]
    )
    arg_parser.add_argument("--mdp-stats-path", type=str, required=True)
    arg_parser.add_argument("--num-stats-episodes", type=int, default=500)
    arg_parser.add_argument(
        "--reward-fn",
        type=str,
        default=SAME_FN,
        choices=[SAME_FN, COUNTERFACTUAL_REWARD_FN],
    )
    arg_parser.add_argument(
        "--control-fn",
        type=str,
        default=QLEARNING,
        choices=[QLEARNING, SARSA],
    )
    arg_parser.add_argument(
        "--intervention-penalty", type=str, default=OFF, choices=(ON, OFF)
    )
    arg_parser.add_argument("--counterfactual-trajectory-prob", type=float, default=0.5)
    arg_parser.add_argument("--output-path", type=str, default=tempfile.gettempdir())
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def run(args: Args, problem_spec: ProblemSpec):
    """
    Runs counterfactual policy experiment.
    """
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Policy Control with Q-learning
    logging.info("Learning counterfactual policy")
    counterfactual_policy, qtable = control(
        control_fn=args.control_fn,
        environment=problem_spec.environment,
        blackbox_policy=problem_spec.blackbox_policy,
        num_episodes=args.num_episodes,
        counterfactual_policy_state_id_fn=create_counterfactual_policy_state_id_fn(
            problem_spec.mdp.env_desc().num_actions,
            state_id_fn=problem_spec.state_id_fn,
        ),
        # counterfactual MDP has |S|x|A| states
        counterfactual_policy_initial_qtable=initial_table(
            num_states=problem_spec.mdp.env_desc().num_states
            * problem_spec.mdp.env_desc().num_actions,
            num_actions=problem_spec.mdp.env_desc().num_actions,
        ),
        epsilon=0.5,
        gamma=1.0,
        alpha=0.1,
        shape_reward_fn=problem_spec.reward_shaping_fn(args.reward_fn),
    )

    logging.info("Collecting stats using blackbox and counterfactual policies")
    logging.info("\n%s", np.around(qtable, 2))

    blackbox_counterfactual_stats = collect_stats(
        problem_spec=problem_spec,
        counterfactual_policy=counterfactual_policy,
        counterfactual_trajectory_prob=args.counterfactual_trajectory_prob,
        num_episodes=args.num_stats_episodes,
    )
    export_stats(blackbox_counterfactual_stats, path=args.output_path)

    problem_spec.environment.close()


def collect_stats(
    problem_spec: ProblemSpec,
    counterfactual_policy: py_policy.PyPolicy,
    counterfactual_trajectory_prob: float,
    num_episodes: int,
):
    """
    Collects statistics from counterfactual trajectories.
    """

    def get_policy_step(
        time_step: ts.TimeStep, policy_state: Any
    ) -> policy_step.PolicyStep:
        bb_policy_step = problem_spec.blackbox_policy.action(time_step, policy_state)
        cf_policy_step = counterfactual_policy.action(time_step, policy_state)
        if bb_policy_step.action != cf_policy_step.action:
            if np.random.rand() <= counterfactual_trajectory_prob:
                return cf_policy_step
        return bb_policy_step

    # stats tracking
    stats = tracking.EpisodeStats()
    blackbox_counterfactual_stats = BlackboxCounterfactualStats(
        blackbox_policy=problem_spec.blackbox_policy,
        counterfactual_policy=counterfactual_policy,
        state_id_fn=problem_spec.state_id_fn,
    )
    time_step = problem_spec.environment.reset()
    policy_state = problem_spec.blackbox_policy.get_initial_state(None)
    episode = 0

    # play N times - use blackbox, log counterfactual policy stats
    while episode < num_episodes:
        bb_or_cf_policy_step = get_policy_step(time_step, policy_state)
        blackbox_counterfactual_stats.accumulate(time_step, policy_state)
        policy_state = bb_or_cf_policy_step.state
        time_step = problem_spec.environment.step(bb_or_cf_policy_step.action)
        stats.new_reward(time_step.reward)

        if time_step.step_type == ts.StepType.LAST:
            episode += 1
            success = problem_spec.success_fn(time_step)
            time_step = problem_spec.environment.reset()
            stats.end_episode(success=success)
            logging.info(str(stats))

    logging.info(
        "Blackbox-Counterfactual stats: \n%s",
        blackbox_counterfactual_stats.state_stats(),
    )
    return blackbox_counterfactual_stats


def parse_problem(
    problem: str, intervention_penalty: str, mdp_stats_path: str
) -> ProblemSpec:
    """
    Parses a problem spec
    """
    if intervention_penalty == ON:
        intervention_reward = 1.0
    elif intervention_penalty == OFF:
        intervention_reward = 0.0
    else:
        raise ValueError(
            f"Unrecognised intervention_penalty: {intervention_penalty}. Must be one of: {ON}, {OFF}"
        )
    if problem == REDGREEN:
        return _parse_redgreen_spec(intervention_reward, mdp_stats_path=mdp_stats_path)
    elif problem == GRIDWORLD:
        return _parse_gridworld_spec(intervention_reward, mdp_stats_path=mdp_stats_path)
    raise ValueError(f"Unsupported problem: {problem}")


def _parse_redgreen_spec(
    intervention_reward: float, mdp_stats_path: str
) -> ProblemSpec:
    arg_parser = argparse.ArgumentParser(prog="RedGreen Params")
    arg_parser.add_argument(
        "--blackbox",
        type=str,
        default=RANDOM,
        choices=[RANDOM, ALWAYS_GREEN, DYNA_PROG],
    )
    arg_parser.add_argument(
        "--cure",
        type=str,
        default="red,green",
    )
    args, _ = arg_parser.parse_known_args()

    cure = args.cure.split(",")
    terminal_state = len(cure)
    env_spec = env_redgreen.create_env_spec(cure=cure)
    mdp = envstats.load_or_generate_inferred_mdp(
        path=mdp_stats_path, env_spec=env_spec, num_episodes=MDP_STATS_EPISODES
    )
    logging.info("Creating blackbox policy %s", args.blackbox)
    if args.blackbox == DYNA_PROG:
        state_values = dynamicprog.iterative_policy_evaluation(
            mdp=mdp,
            policy=policies.PyRandomObservablePolicy(
                time_step_spec=env_spec.environment.time_step_spec(),
                action_spec=env_spec.environment.action_spec(),
                num_actions=env_spec.env_desc.num_actions,
            ),
        )
        action_values = dynamicprog.action_values_from_state_values(
            mdp=mdp, state_values=state_values
        )
        blackbox_policy = policies.PyQGreedyPolicy(
            time_step_spec=env_spec.environment.time_step_spec(),
            action_spec=env_spec.environment.action_spec(),
            state_id_fn=env_redgreen.get_state_id,
            action_values=action_values,
        )

    elif args.blackbox == ALWAYS_GREEN:
        action_values = np.zeros(
            shape=(mdp.env_desc().num_states, mdp.env_desc().num_actions),
            dtype=np.float32,
        )
        # green is the best
        action_values[:, constants_redgreen.GREEN_PILL] = 1.0
        blackbox_policy = policies.PyQGreedyPolicy(
            time_step_spec=env_spec.environment.time_step_spec(),
            action_spec=env_spec.environment.action_spec(),
            state_id_fn=env_redgreen.get_state_id,
            action_values=action_values,
        )
    else:
        blackbox_policy = policies.PyRandomPolicy(
            time_step_spec=env_spec.environment.time_step_spec(),
            action_spec=env_spec.environment.action_spec(),
            num_actions=env_spec.env_desc.num_actions,
        )

    def success_fn(time_step: ts.TimeStep):
        return (
            time_step.observation[constants_redgreen.KEY_OBS_POSITION] == terminal_state
        )

    def create_reward_shaping_fns() -> Mapping[str, rewards.RewardShapeFn]:
        def counterfactual_reward_fn(next_time_step: ts.TimeStep) -> float:
            distance_to_end = (
                len(
                    next_time_step.observation[constants_redgreen.KEY_OBS_CURE_SEQUENCE]
                )
                - next_time_step.observation[constants_redgreen.KEY_OBS_POSITION]
            )
            # closer is better
            return next_time_step.reward - distance_to_end

        reward_fns = {
            SAME_FN: rewards.create_identity_reward_fn(),
            COUNTERFACTUAL_REWARD_FN: rewards.create_time_step_shaped_reward_fn(
                counterfactual_reward_fn
            ),
        }
        return {
            key: rewards.apply_intervention_reward(reward_fn, intervention_reward)
            for key, reward_fn in reward_fns.items()
        }

    return ProblemSpec(
        environment=env_spec.environment,
        mdp=mdp,
        blackbox_policy=blackbox_policy,
        state_id_fn=env_redgreen.get_state_id,
        success_fn=success_fn,
        reward_shaping_functions=create_reward_shaping_fns,
    )


def _parse_gridworld_spec(
    intervention_reward: float, mdp_stats_path: str
) -> ProblemSpec:
    arg_parser = argparse.ArgumentParser(prog="GridWorld Params")
    arg_parser.add_argument(
        "--grid-path",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.pardir,
            os.path.pardir,
            os.path.pardir,
            "assets",
            "env",
            "gridworld",
            "levels",
            "gridworld_cliff_02.txt",
        ),
    )
    arg_parser.add_argument(
        "--blackbox",
        type=str,
        default=RANDOM,
        choices=[RANDOM, DYNA_PROG, QLEARNING],
    )
    args, _ = arg_parser.parse_known_args()

    grid_size, cliffs, exits, start = utils_gridworld.parse_grid(path=args.grid_path)
    env_spec = env_gridworld.create_env_spec(
        grid_size, cliffs=cliffs, exits=exits, start=start
    )
    mdp = envstats.load_or_generate_inferred_mdp(
        path=mdp_stats_path, env_spec=env_spec, num_episodes=MDP_STATS_EPISODES
    )
    state_id_fn = env_gridworld.create_state_id_fn(
        env_gridworld.states_mapping(size=grid_size, cliffs=cliffs)
    )
    logging.info("Creating blackbox policy %s", args.blackbox)
    if args.blackbox == DYNA_PROG:
        # learns a safe policy - keey away from dangerous cliffs
        state_values = dynamicprog.iterative_policy_evaluation(
            mdp=mdp,
            policy=policies.PyRandomObservablePolicy(
                time_step_spec=env_spec.environment.time_step_spec(),
                action_spec=env_spec.environment.action_spec(),
                num_actions=env_spec.env_desc.num_actions,
            ),
        )
        action_values = dynamicprog.action_values_from_state_values(
            mdp=mdp, state_values=state_values
        )
        blackbox_policy = policies.PyQGreedyPolicy(
            time_step_spec=env_spec.environment.time_step_spec(),
            action_spec=env_spec.environment.action_spec(),
            state_id_fn=state_id_fn,
            action_values=action_values,
        )
    elif args.blackbox == QLEARNING:
        blackbox_policy, _ = qlearning.control(
            environment=env_spec.environment,
            num_episodes=5000,
            log_step=1000,
            state_id_fn=state_id_fn,
            initial_qtable=np.zeros(
                shape=(env_spec.env_desc.num_states, env_spec.env_desc.num_actions)
            ),
            epsilon=0.1,
            gamma=1.0,
            alpha=0.1,
        )
    else:
        blackbox_policy = policies.PyRandomPolicy(
            time_step_spec=env_spec.environment.time_step_spec(),
            action_spec=env_spec.environment.action_spec(),
            num_actions=env_spec.env_desc.num_actions,
        )

    def success_fn(time_step: ts.TimeStep):
        return (
            time_step.observation[constants_gridworld.Strings.player]
            in time_step.observation[constants_gridworld.Strings.exits]
        )

    def create_reward_shaping_fns() -> Mapping[str, rewards.RewardShapeFn]:
        def counterfactual_reward_fn(next_time_step: ts.TimeStep) -> float:
            distances = []
            player = np.array(
                next_time_step.observation[constants_gridworld.Strings.player], np.float
            )
            for cliff in next_time_step.observation[constants_gridworld.Strings.cliffs]:
                distances.append(np.linalg.norm(player - np.array(cliff, np.float)))
            return next_time_step.reward - (
                1.0 / np.min(distances) if len(distances) > 0 else 0.0
            )

        reward_fns = {
            SAME_FN: rewards.create_identity_reward_fn(),
            COUNTERFACTUAL_REWARD_FN: rewards.create_time_step_shaped_reward_fn(
                counterfactual_reward_fn
            ),
        }
        return {
            key: rewards.apply_intervention_reward(reward_fn, intervention_reward)
            for key, reward_fn in reward_fns.items()
        }

    return ProblemSpec(
        environment=env_spec.environment,
        mdp=mdp,
        blackbox_policy=blackbox_policy,
        state_id_fn=state_id_fn,
        success_fn=success_fn,
        reward_shaping_functions=create_reward_shaping_fns(),
    )


def control(
    control_fn: str,
    environment: py_environment.PyEnvironment,
    blackbox_policy: policies.PyPolicy,
    num_episodes: int,
    counterfactual_policy_state_id_fn: Callable[[NestedArray, NestedArray], int],
    counterfactual_policy_initial_qtable: np.ndarray,
    epsilon: float,
    gamma: float,
    alpha: float,
    shape_reward_fn: rewards.RewardShapeFn,
    log_steps: int = 100,
) -> Tuple[py_policy.PyPolicy, np.ndarray]:
    """
    Counterfactual policy control.
    Implements Q-learning and SARSA, using epsilon-greedy as a collection (behavior) policy.
    Source: https://homes.cs.washington.edu/~bboots/RL-Spring2020/Lectures/TD_notes.pdf
    """
    if control_fn == QLEARNING:
        _control_fn = _qlearing_step
    elif control_fn == SARSA:
        _control_fn = _sarsa_step
    else:
        raise ValueError(f"Unknown control-fn {control_fn}")

    return _control(
        control_fn=_control_fn,
        environment=environment,
        blackbox_policy=blackbox_policy,
        num_episodes=num_episodes,
        counterfactual_policy_state_id_fn=counterfactual_policy_state_id_fn,
        counterfactual_policy_initial_qtable=counterfactual_policy_initial_qtable,
        epsilon=epsilon,
        gamma=gamma,
        alpha=alpha,
        shape_reward_fn=shape_reward_fn,
        log_steps=log_steps,
    )


def _control(
    control_fn: Callable[
        [
            np.ndarray,
            policies.PyPolicy,
            Callable[[NestedArray, NestedArray], int],
            float,
            float,
            Sequence[trajectory.Trajectory],
        ],
        np.ndarray,
    ],
    environment: py_environment.PyEnvironment,
    blackbox_policy: policies.PyPolicy,
    num_episodes: int,
    counterfactual_policy_state_id_fn: Callable[[NestedArray, NestedArray], int],
    counterfactual_policy_initial_qtable: np.ndarray,
    epsilon: float,
    gamma: float,
    alpha: float,
    shape_reward_fn: rewards.RewardShapeFn,
    log_steps: int,
) -> Tuple[py_policy.PyPolicy, np.ndarray]:
    """
    Counterfactual policy control.
    Implements Q-learning and SARSA, using epsilon-greedy as a collection (behavior) policy.
    Source: https://homes.cs.washington.edu/~bboots/RL-Spring2020/Lectures/TD_notes.pdf
    """
    qtable = copy.deepcopy(counterfactual_policy_initial_qtable)
    policy, collect_policy = _counterfactual_target_and_collect_policies(
        blackbox_policy=blackbox_policy,
        counterfactual_policy_state_id_fn=counterfactual_policy_state_id_fn,
        qtable=qtable,
        epsilon=epsilon,
    )
    episode = 0
    while episode < num_episodes:
        environment.reset()
        step = 0
        transitions = []
        while True:
            time_step = environment.current_time_step()
            policy_step = collect_policy.action(time_step)
            _next_time_step = environment.step(policy_step.action)
            next_time_step = shape_reward_fn(
                record.BlackboxCounterfactual(
                    blackbox_policy=policy, counterfactual_policy=blackbox_policy
                ),
                record.Step(
                    time_step=time_step,
                    policy_step=policy_step,
                    next_time_step=_next_time_step,
                ),
                (),
            )
            traj = trajectory.from_transition(time_step, policy_step, next_time_step)
            transitions.append(traj)
            step += 1

            if len(transitions) == 2:
                # modifies qtable
                qtable = control_fn(
                    qtable,
                    blackbox_policy,
                    counterfactual_policy_state_id_fn,
                    gamma=gamma,
                    alpha=alpha,
                    experiences=transitions,
                )
                # update policies
                policy, collect_policy = _counterfactual_target_and_collect_policies(
                    blackbox_policy=blackbox_policy,
                    counterfactual_policy_state_id_fn=counterfactual_policy_state_id_fn,
                    qtable=qtable,
                    epsilon=epsilon,
                )

                # remove earliest step
                transitions.pop(0)

            if time_step.step_type == ts.StepType.LAST:
                break

        episode += 1
        if episode % log_steps == 0:
            logging.info(
                "Trained with %d steps, episode %d/%d",
                step,
                episode,
                num_episodes,
            )
    return policy, qtable


def _qlearing_step(
    qtable: np.ndarray,
    blackbox_policy: policies.PyPolicy,
    counterfactual_policy_state_id_fn: Callable[[NestedArray, NestedArray], int],
    gamma: float,
    alpha: float,
    experiences: Sequence[trajectory.Trajectory],
) -> np.ndarray:
    """
    Q-learning update step.
    """
    steps = len(experiences)
    new_qtable = copy.deepcopy(qtable)

    if steps < 2:
        logging.warning("Q-learning requires at least two steps per update - skipping")
        return new_qtable
    for step in range(steps - 1):
        state_id = counterfactual_policy_state_id_fn(
            experiences[step].observation,
            blackbox_policy.action(time_step=experiences[step]).action,
        )
        next_state_id = counterfactual_policy_state_id_fn(
            experiences[step + 1].observation,
            blackbox_policy.action(time_step=experiences[step + 1]).action,
        )

        state_action_value = qtable[state_id, experiences[step].action]
        next_state_actions_values = qtable[next_state_id]
        next_best_action = np.random.choice(
            np.flatnonzero(next_state_actions_values == next_state_actions_values.max())
        )

        delta = (
            experiences[step].reward + gamma * qtable[next_state_id, next_best_action]
        ) - state_action_value
        state_action_value = state_action_value + alpha * delta
        new_qtable[state_id, experiences[step].action] = state_action_value
    return new_qtable


def _sarsa_step(
    qtable: np.ndarray,
    blackbox_policy: policies.PyPolicy,
    counterfactual_policy_state_id_fn: Callable[[NestedArray, NestedArray], int],
    gamma: float,
    alpha: float,
    experiences: Sequence[trajectory.Trajectory],
) -> np.ndarray:
    """
    SARSA update step.
    """
    steps = len(experiences)
    new_qtable = copy.deepcopy(qtable)

    if steps < 2:
        logging.warning("SARSA requires at least two steps per update - skipping")
        return new_qtable
    for step in range(steps - 1):
        state_id = counterfactual_policy_state_id_fn(
            experiences[step].observation,
            blackbox_policy.action(time_step=experiences[step]).action,
        )
        next_state_id = counterfactual_policy_state_id_fn(
            experiences[step + 1].observation,
            blackbox_policy.action(time_step=experiences[step + 1]).action,
        )

        state_action_value = qtable[state_id, experiences[step].action]
        next_state_action_value = qtable[next_state_id, experiences[step + 1].action]
        delta = (
            experiences[step].reward + gamma * next_state_action_value
        ) - state_action_value
        state_action_value = state_action_value + alpha * delta
        new_qtable[state_id, experiences[step].action] = state_action_value
    return new_qtable


def _counterfactual_target_and_collect_policies(
    blackbox_policy: policies.PyPolicy,
    counterfactual_policy_state_id_fn: Callable[[NestedArray, NestedArray], int],
    qtable: np.ndarray,
    epsilon: float,
) -> Tuple[py_policy.PyPolicy, py_policy.PyPolicy]:
    _, num_actions = qtable.shape
    policy = CounterfactualQGreedyPolicy(
        blackbox_policy=blackbox_policy,
        state_id_fn=counterfactual_policy_state_id_fn,
        action_values=qtable,
    )
    collect_policy = policies.PyEpsilonGreedyPolicy(
        policy=policy,
        num_actions=num_actions,
        epsilon=epsilon,
    )
    return policy, collect_policy


def export_stats(
    blackbox_counterfactual_stats: BlackboxCounterfactualStats, path: str
) -> None:
    """
    Exports stats to a json file.
    """
    logs = sorted(
        [
            (state, entry)
            for state, entry in blackbox_counterfactual_stats.state_stats().items()
        ],
        key=lambda xy: xy[0],
    )

    with open(
        os.path.join(path, "blackbox_counterfactual_stats.json"),
        "w",
        encoding="UTF-8",
    ) as writable:
        for state, entry in logs:
            payload = serialize({"state": state, "stats": entry})
            writable.write(json.dumps(payload))
            writable.write("\n")


def initial_table(num_states: int, num_actions: int) -> np.ndarray:
    """
    Initializes a Q-table to zeros.
    Returns:
        A numpy array with dimensions `num_states, num_actions`.
    """
    return np.zeros(shape=(num_states, num_actions))


def create_counterfactual_policy_state_id_fn(
    num_actions: int, state_id_fn: Callable[[NestedArray], int]
) -> Callable[[NestedArray, NestedArray], int]:
    """
    Creates a function that maps observations and blackbox policy actions
    into a state ID for the counterfactual policy.

    Returns
    """

    def get_state_id(observation: NestedArray, blackbox_policy_action: NestedArray):
        env_state_id = state_id_fn(observation)
        return blackbox_policy_action + (num_actions * env_state_id)

    return get_state_id


def serialize(obj: Any) -> Mapping[str, Any]:
    """
    Serialize payload.
    """
    if isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, list):
        return [serialize(x) for x in obj]
    elif isinstance(obj, set):
        return {serialize(x) for x in obj}
    elif isinstance(obj, tuple):
        return (serialize(x) for x in obj)
    elif isinstance(obj, dict):
        return {str(key): serialize(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {
            str(key): serialize(value)
            for key, value in getattr(obj, "__dict__").items()
        }
    return obj


def main():
    args = parse_args()
    problem_spec = parse_problem(
        problem=args.problem,
        intervention_penalty=args.intervention_penalty,
        mdp_stats_path=args.mdp_stats_path,
    )
    run(args, problem_spec)


if __name__ == "__main__":
    main()
