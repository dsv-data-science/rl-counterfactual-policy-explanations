"""
Module contains job to collect transition statistics from environments.
"""


import argparse
import collections
import dataclasses
import json
import logging
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import ray
from rlplg.learning.tabular import empiricmdp

import daaf
from daaf import utils
from daaf.envstats import envstats


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Program arguments.
    """

    # problem args
    config_path: str
    num_episodes: int
    output_dir: str
    # ray args
    cluster_uri: Optional[str]
    num_workers: int


@dataclasses.dataclass(frozen=True)
class TaskArgs:
    """
    Task config for supervisor.
    """

    env_name: str
    num_episodes: int
    env_args: Mapping[str, Any]
    logging_frequency_episodes: int


@ray.remote
def work(task_id: str, task_args: TaskArgs):
    """
    Collects stats for an env.
    """
    logging.info("Worker %s starting", task_id)
    return envstats.collect_stats(
        env_name=task_args.env_name,
        num_episodes=task_args.num_episodes,
        env_args=task_args.env_args,
        logging_frequency_episodes=task_args.logging_frequency_episodes,
    )


def main(args: Args):
    """
    Entry point.
    """
    ray_env = {
        "py_modules": [daaf],
    }
    logging.info("Ray environment: %s", ray_env)
    with open(args.config_path, encoding="UTF-8") as readable:
        env_configs = json.load(readable)

    with ray.init(args.cluster_uri, runtime_env=ray_env) as context:
        logging.info("Ray Context: %s", context)
        logging.info("Ray Nodes: %s", ray.nodes())

        logging_frequency_episodes = 1000
        tasks = create_tasks(
            env_configs,
            num_episodes=args.num_episodes,
            num_workers=args.num_workers,
            logging_frequency_episodes=logging_frequency_episodes,
        )

    # run without backpressure
    result_refs: List[ray.ObjectRef] = []
    results: Sequence[Tuple[Any, Any]] = []
    for task_id, task_args in tasks:
        # Allow  args.num_workers in flight calls
        if len(result_refs) > args.num_workers:
            # update result_refs to only
            # track the remaining tasks.
            num_ready = len(result_refs) - args.num_workers
            results, inflight_result_refs = wait(
                results, result_refs, num_ready=num_ready
            )
            result_refs = list(inflight_result_refs)
        result_refs.append(work.remote(task_id, task_args))
    results, _ = wait(results, result_refs, num_ready=len(result_refs))
    output = aggretate_results(results)
    envstats.export_stats(args.output_dir, output)
    logging.info("Done!")


def create_tasks(
    env_configs: Mapping[str, Any],
    num_episodes: int,
    num_workers: int,
    logging_frequency_episodes: int,
) -> Sequence[Tuple[str, TaskArgs]]:
    """
    Creates configuration for tasks.
    Every environment config has their episodes split across num_workers -
    if there are more workers than episodes, only tasks for the necessary
    number of workers is created.
    """
    tasks = []
    worker_splits = tuple(
        utils.split(items=tuple(range(num_episodes)), num_partitions=num_workers)
    )
    for (env_name, configs) in env_configs.items():
        for config in configs:
            env_args = config["args"]
            # for each config, split episodes between workers
            for idx, worker_split in enumerate(worker_splits):
                task_args = TaskArgs(
                    env_name=env_name,
                    num_episodes=len(worker_split),
                    env_args=env_args,
                    logging_frequency_episodes=logging_frequency_episodes,
                )
                tasks.append((f"{env_name}-{idx}", task_args))
    return tuple(tasks)


def wait(
    results: Sequence[Any], result_refs: Sequence[ray.ObjectRef], num_ready: int
) -> Tuple[Sequence[Any], Sequence[ray.ObjectRef]]:
    """
    Wait for tasks and update lists
    """
    newly_completed, result_refs = ray.wait(result_refs, num_returns=num_ready)
    new_results = []
    for completed_ref in newly_completed:
        new_results.append(ray.get(completed_ref))
    return list(results) + new_results, result_refs


def aggretate_results(results: Sequence[Tuple[Any, Any]]) -> Sequence[Tuple[Any, Any]]:
    """
    Aggregates results by env-level (key).
    """
    results_agg = collections.defaultdict(list)
    for env_level, mdp_stats in results:
        results_agg[env_level].append(mdp_stats)
    output = []
    for env_level in results_agg:
        output.append((env_level, empiricmdp.aggregate_stats(results_agg[env_level])))
    return output


def parse_args() -> Args:
    """
    Parses program arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-path", type=str, required=True)
    arg_parser.add_argument("--num-episodes", type=int, required=True)
    arg_parser.add_argument("--output-dir", type=str, required=True)
    arg_parser.add_argument("--cluster-uri", type=str, default=None)
    arg_parser.add_argument("--num-workers", type=int, default=1)
    known_args, unknown_args = arg_parser.parse_known_args()
    logging.info("Unknown args: %s", unknown_args)
    return Args(**vars(known_args))


if __name__ == "__main__":
    main(parse_args())
