"""
Module contains utitilies to collect transition statistics from an environment.
"""


import collections
import logging
import os.path
from typing import Any, Mapping, Optional, Sequence, Tuple

import tensorflow as tf
from rlplg import envspec, envsuite
from rlplg.learning.tabular import empiricmdp, policies

EnvLevel = collections.namedtuple("EnvLevel", ["name", "level"])


def collect_stats(
    env_name: str,
    num_episodes: int,
    env_args: Mapping[str, Any],
    logging_frequency_episodes: int,
) -> Tuple[EnvLevel, empiricmdp.MdpStats]:
    """
    Loads and env and aggreates trajectory data.
    """
    logging.info(
        "Running stats on %d episodes for %s with args %s",
        num_episodes,
        env_name,
        env_args,
    )
    env_spec = envsuite.load(env_name, **env_args)
    mdp_stats = generate_mdp_stats(
        env_spec=env_spec,
        num_episodes=num_episodes,
        logging_frequency_episodes=logging_frequency_episodes,
    )
    return EnvLevel(env_spec.name, env_spec.level), mdp_stats


def load_or_generate_mdp_stats(
    path: str,
    env_spec: envspec.EnvSpec,
    mdp_stats_num_episodes: Optional[int] = None,
    logging_frequency_episodes: int = 10,
) -> empiricmdp.MdpStats:
    """
    Loads mdp stats for an environment if it is found in the given path
    and computes it otherwise is num_episodes is given.
    """
    try:
        stats_filename = create_stats_filename(
            env_name=env_spec.name, level=env_spec.level
        )
        mdp_stats = empiricmdp.load_stats(path=path, filename=stats_filename)
        logging.info("Loaded stats from %s/%s", path, stats_filename)
        return mdp_stats
    except IOError as ioerr:
        if mdp_stats_num_episodes is not None:
            logging.info(
                "Failed to load stats from %s/%s. Computing stats with %d episodes",
                path,
                stats_filename,
                mdp_stats_num_episodes,
            )
            mdp_stats = generate_mdp_stats(
                env_spec=env_spec,
                num_episodes=mdp_stats_num_episodes,
                logging_frequency_episodes=logging_frequency_episodes,
            )
            empiricmdp.export_stats(
                path=path, filename=stats_filename, mdp_stats=mdp_stats
            )
            return mdp_stats
        logging.info(
            "Failed to load stats from %s/%s. Num episodes is None. Returning None.",
            path,
            stats_filename,
        )
        raise ValueError(
            f"Failed to load stats from {path}/{stats_filename}. Num episodes is None. Either provide a valid path or num of episodes to generate stats.",
        ) from ioerr


def load_or_generate_inferred_mdp(
    path: str, env_spec: envspec.EnvSpec, num_episodes: Optional[int] = None
) -> Optional[empiricmdp.InferredMdp]:
    """
    Loads mdp stats for an environment if it is found in the given path
    and computes it otherwise is num_episodes is given.
    """
    mdp_stats = load_or_generate_mdp_stats(
        path=path, env_spec=env_spec, mdp_stats_num_episodes=num_episodes
    )

    mdp_functions = empiricmdp.create_mdp_functions(mdp_stats)
    return empiricmdp.InferredMdp(
        mdp_functions=mdp_functions, env_desc=env_spec.env_desc
    )


def generate_mdp_stats(
    env_spec: envspec.EnvSpec, num_episodes: int, logging_frequency_episodes: int
) -> empiricmdp.MdpStats:
    """
    Computes MDP stats for an environment using a random policy.
    """
    policy = policies.PyRandomPolicy(
        time_step_spec=env_spec.environment.time_step_spec(),
        action_spec=env_spec.environment.action_spec(),
        num_actions=env_spec.env_desc.num_actions,
    )
    return empiricmdp.collect_mdp_stats(
        environment=env_spec.environment,
        policy=policy,
        state_id_fn=env_spec.discretizer.state,
        action_id_fn=env_spec.discretizer.action,
        num_episodes=num_episodes,
        logging_frequency_episodes=logging_frequency_episodes,
    )


def create_stats_filename(env_name: str, level: str = None) -> str:
    """
    Generates a filename given an environment and level.

    Returns:
        A unique file name for an env and level.
    """
    if level is None:
        return env_name
    return f"{env_name}-{level}"


def export_stats(
    output_path: str, results: Sequence[Tuple[EnvLevel, empiricmdp.MdpStats]]
):
    """
    Export stats from multiple environments.
    """
    for env_level, mdp_stats in results:
        stats_path = os.path.join(output_path, env_level.name, "stats")
        if not tf.io.gfile.exists(stats_path):
            tf.io.gfile.makedirs(stats_path)
        logging.info("Exporting stats for %s", env_level.name)
        empiricmdp.export_stats(
            stats_path,
            create_stats_filename(env_level.name, env_level.level),
            mdp_stats,
        )
        _ = empiricmdp.load_stats(
            stats_path, create_stats_filename(env_level.name, env_level.level)
        )
        logging.info("Exported stats for %s/%s", env_level.name, env_level.level)
