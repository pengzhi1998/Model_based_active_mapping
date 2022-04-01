"""benchmarks.py
Run benchmarks for frontier based exploration
"""
from __future__ import print_function, absolute_import, division

import multiprocessing
import os

import numpy as np
import pickle

from bc_exploration.algorithms.frontier_based_exploration import run_frontier_exploration
from bc_exploration.utilities.paths import get_maps_dir, get_exploration_dir


def make_results(result_keys, results):
    """
    Makes a result list from output results and a list of keys
    :param result_keys Iterable: keys in order for each result
    :param results Iterable: matching data for the keys
    :return List[dict]: result list from output results and a list of keys
    """
    return [dict(zip(result_keys, result)) for result in results]


def make_batches(dataset, batch_size):
    """
    Returns a generator object that will yield batches of the dataset specified,
    assuming the data is directly iterable by a single index
    :param dataset Iterable: data to batch, dataset[i] should separate by data point
    :param batch_size int: the size of which to batch
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def run_one_frontier(config, return_queue):
    """
    Given a config, run the frontier agent with those params, and put the results in the results queue
    :param config dict: corresponding to all the function params to run_frontier_exploration(...)
    :param return_queue multiprocessing.Queue:, containing the results for all the processes created
    """
    # occupancy map cannot be returned, process.join() will hang
    _, percent_explored, iterations_taken, was_sucessful = run_frontier_exploration(**config)
    return_queue.put(dict(zip(['percent_explored', 'iterations_taken', 'was_successful'],
                              (percent_explored, iterations_taken, was_sucessful))))


def run_frontier_benchmark(config, num_instances, num_processes):
    """
    Runs num_instances runs with config, multiprocesses based off of num_processes given
    :param config dict: corresponding to all the function params to run_frontier_exploration(...)
    :param num_instances int: total number of runs to do
    :param num_processes int: number of runs to do at one time (processes used)
    :return List[dict]: results from all the runs
    """
    result_queue = multiprocessing.Queue()

    compute_groups = list(make_batches(list(range(0, num_instances)), num_processes))
    for compute_group in compute_groups:
        processes = []
        for process_idx in compute_group:
            np.random.seed(process_idx)
            process = multiprocessing.Process(target=run_one_frontier, args=(config, result_queue,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    results = []
    for i in range(num_instances):
        result = result_queue.get().copy()
        result.update({'seed': i})
        results.append(result)

    return results


def run_consistency(frontier_config, num_instances):
    """
    Runs the consistency test, i.e the exploration algothirm is mostly deterministic,
    should it should return very similar results
    :param frontier_config dict: corresponding to all the function params to run_frontier_exploration(...)
    :param num_instances int: total number of runs to do
    :return List[dict]: results from the benchmark
    """
    consistency_results = run_frontier_benchmark(frontier_config, num_instances=num_instances, num_processes=4)
    iterations = [consistency_result['iterations_taken'] for consistency_result in consistency_results]
    assert np.std(iterations) < 3
    return consistency_results


def run_completion(frontier_config, num_instances):
    """
     Runs the completion test, i.e the exploration algorithm is run from multiple random locations in the same
     environment and expected to solve it every run
     :param frontier_config dict: corresponding to all the function params to run_frontier_exploration(...)
     :param num_instances int: total number of runs to do
     :return List[dict]: results from the benchmark
     """
    completion_results = run_frontier_benchmark(frontier_config, num_instances=num_instances, num_processes=4)
    successes = [completion_result['was_successful'] for completion_result in completion_results]
    assert np.all(successes)
    return completion_results


def run_frontier_benchmarks():
    """
    Runs the frontier benchmarks
    """
    # shared parameters
    num_instances = 20
    sensor_range = 10.0
    completion_percentage = 0.95
    max_exploration_iterations = 75
    params = "params/params.yaml"

    # individual parameters
    maps_to_run = ["brain/vw_ground_truth_full_edited.png"]
    consistency_start_states = [np.array([2.5, 5.5, -np.pi / 4])]

    results = {}
    for i, map_name in enumerate(maps_to_run):
        map_filename = os.path.join(get_maps_dir(), map_name)
        params_filename = os.path.join(get_exploration_dir(), params)

        completion_config = dict(map_filename=map_filename,
                                 params_filename=params_filename,
                                 start_state=None,
                                 sensor_range=sensor_range,
                                 completion_percentage=completion_percentage,
                                 render=False,
                                 render_wait_for_key=False,
                                 max_exploration_iterations=max_exploration_iterations)

        consistency_config = completion_config.copy()
        consistency_config['start_state'] = consistency_start_states[i]

        consistency_results = run_consistency(consistency_config, num_instances=num_instances)
        completion_results = run_completion(completion_config, num_instances=num_instances)

        print(map_name)
        print('consistency:')
        _ = [print(consistency_result) for consistency_result in consistency_results]
        print()
        print('completion:')
        _ = [print(completion_result) for completion_result in completion_results]
        print()

        results[map_name] = {'consistency': consistency_results,
                             'completion': completion_results}

    with open('benchmark_results.pkl', 'w') as f:
        pickle.dump(results, f)


def main():
    """
    Main function
    """
    run_frontier_benchmarks()


if __name__ == '__main__':
    main()
