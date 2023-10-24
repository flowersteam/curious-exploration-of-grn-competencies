from autodiscjax import DictTree
from autodiscjax.experiment_pipelines import run_robustness_tests
from autodiscjax.utils.create_modules import *
from autodiscjax.utils.timeseries import is_stable
import csv
import evaluation_config
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import os
from shapely import Point, Polygon, unary_union

def select_representants(experiment_system_output_library, observed_node_ids,
                         stable_test_n_last_steps, stable_test_settling_threshold,
                         n_representants, epsilon, n_trials, seed, save_folder):
    representants = DictTree()

    trajectories = experiment_system_output_library.ys[:, jnp.array(observed_node_ids)]
    is_valid_bool = ~(jnp.isnan(trajectories).any(-1).any(-1)) & (trajectories>=-1e-6).all(-1).all(-1)
    is_stable_bool, _, _ = is_stable(trajectories,
                                     time_window=jnp.r_[-stable_test_n_last_steps:0],
                                     settling_threshold=stable_test_settling_threshold)
    is_stable_bool = is_stable_bool.all(-1)
    preselected_ids = jnp.where(is_valid_bool & is_stable_bool)[0]


    reached_endpoints = trajectories[preselected_ids, :, -1]
    reached_space_min = jnp.nanmin(reached_endpoints, axis=0)
    reached_space_max = jnp.nanmax(reached_endpoints, axis=0)
    normalized_endpoints = (reached_endpoints-reached_space_min) / (reached_space_max-reached_space_min)

    total_union_polygon = unary_union([Point(reached_endpoint).buffer(epsilon) for reached_endpoint in normalized_endpoints])
    total_covered_area = total_union_polygon.area

    max_covered_ratio = 0.
    key = jrandom.PRNGKey(seed)
    for _ in range(n_trials):
        key, subkey = jrandom.split(key)
        sample_ids = jrandom.choice(subkey, jnp.arange(len(reached_endpoints)), shape=(n_representants,), replace=False)
        union_polygon = unary_union([Point(reached_endpoint).buffer(epsilon) for reached_endpoint in normalized_endpoints[sample_ids]])
        covered_ratio = union_polygon.area / total_covered_area

        if covered_ratio > max_covered_ratio:
            max_covered_ratio = covered_ratio
            representative_ids = sample_ids

    representants.representative_ids = preselected_ids[representative_ids]
    representants.covered_ratio = max_covered_ratio

    representants.save(os.path.join(save_folder, "representants.pickle"), overwrite=True)

    return representants.representative_ids


def calc_settling_time(dist_vals, settling_time_threshold):
    # assume normalized dist_vals starting from 1 and finishing at 0
    settling_time = jnp.where(~(dist_vals < settling_time_threshold), size=len(dist_vals), fill_value=-1)[0].max()
    return settling_time


def calc_travelling_time(trajectories):
    distance_travelled = jnp.cumsum(jnp.sqrt(jnp.sum(jnp.diff(trajectories, axis=-1) ** 2, axis=-2)), axis=-1)
    distance_travelled = distance_travelled / distance_travelled.max(-1)  # normalize between 0 and 1
    T10 = jnp.where(distance_travelled >= 0.1, size=distance_travelled.shape[-1], fill_value=-1)[0][0]
    T90 = jnp.where(distance_travelled >= 0.9, size=distance_travelled.shape[-1], fill_value=-1)[0][0]
    return T10, T90


def calc_trajectories_statistics(trajectories, settling_time_threshold, save_folder):
    trajectories = trajectories[..., 1:]  # remove first step (dont wanna take into account big jumps happening in first step)

    # settling time:  first time T such that the distance between y(t) and yfinal ≤ 0.02 × |yfinal – yinit| for t ≥ T
    # normalize such that origin is final point and unit=(end-origin)
    trajectories_extent = (trajectories.max(-1) - trajectories.min(-1))[..., jnp.newaxis]
    trajectories_extent = trajectories_extent.at[trajectories_extent == 0.].set(1.)
    normalized_trajectories = (trajectories - trajectories[..., -1][..., jnp.newaxis]) / trajectories_extent
    distance_to_target = jnp.linalg.norm(normalized_trajectories, axis=1)
    distance_to_target = distance_to_target / distance_to_target[:, 0][:, jnp.newaxis]
    settling_times = vmap(calc_settling_time, in_axes=(0, None))(distance_to_target, settling_time_threshold)

    # travelling time: time it takes for the response to travel from 10% to 90% of the way from yinit to yfinal
    T10s, T90s = vmap(calc_travelling_time)(normalized_trajectories)

    # detours (duration and area)
    detours_duration = []
    detours_area = []
    detours_timesteps = []

    for sample_idx in range(len(distance_to_target)):
        detour_timesteps = []
        detour_duration = 0.
        detour_area = 0.
        if settling_times[sample_idx] > 0:
            cur_distance_to_target = distance_to_target[sample_idx, :settling_times[sample_idx]]
            is_distance_increasing = jnp.concatenate(
                [jnp.array([False]), jnp.diff(cur_distance_to_target) > 1e-3 * cur_distance_to_target.max()])
            is_distance_decreasing = jnp.concatenate(
                [jnp.array([True]), jnp.diff(cur_distance_to_target) < 1e-3 * cur_distance_to_target.max()])
            start_detour_timesteps = jnp.where(is_distance_decreasing[:-1] & is_distance_increasing[1:])[0]
            if len(start_detour_timesteps) > 0:
                start_detour_dist_vals = cur_distance_to_target[start_detour_timesteps]
                end_detour_timesteps = []

                for start_detour_timestep, start_detour_dist_val in zip(start_detour_timesteps, start_detour_dist_vals):
                    possible_detour_timesteps = jnp.where((cur_distance_to_target[:-1] >= start_detour_dist_val) &
                                                          (cur_distance_to_target[1:] <= start_detour_dist_val))[0] + 1
                    # take the first time step (after start_detour_timestep) where distance curve is crossing back y=start_detour_dist_val
                    # if no crossing back before settling time, we consider settling time as the end of the detour
                    possible_end_detour_timesteps = possible_detour_timesteps[
                        possible_detour_timesteps > start_detour_timestep]
                    if len(possible_end_detour_timesteps) > 0:
                        end_detour_timestep = possible_end_detour_timesteps[0]
                    else:
                        end_detour_timestep = settling_times[sample_idx] - 1
                    end_detour_timesteps.append(end_detour_timestep)

                # calc union of intervals (in case some overlaps due to noise)
                detour_timesteps = jnp.where(jnp.array([(jnp.arange(len(cur_distance_to_target)) >= start) &
                                                        (jnp.arange(len(cur_distance_to_target)) <= end)
                                                        for (start, end) in
                                                        zip(start_detour_timesteps, end_detour_timesteps)]).any(0))[0]
                detour_duration = len(detour_timesteps)
                # detour_area = jnp.minimum(cur_distance_to_target[detour_timesteps[:-1]],
                #                           cur_distance_to_target[detour_timesteps[1:]]) * 1
                # detour_area += (jnp.maximum(cur_distance_to_target[detour_timesteps[:-1]],
                #                             cur_distance_to_target[detour_timesteps[1:]]) * 1 - detour_area) / 2.
                # detour_area = detour_area.sum()
                rel_start_detours_timesteps = jnp.concatenate(
                    [jnp.array([0]), jnp.where((detour_timesteps[1:] - detour_timesteps[:-1]) > 1)[0] + 1])
                rel_end_detours_timesteps = jnp.roll(rel_start_detours_timesteps - 1, -1)

                detour_polygon = Polygon()
                for start, end in zip(rel_start_detours_timesteps, rel_end_detours_timesteps):
                    detour_points = normalized_trajectories[sample_idx][:, detour_timesteps[start:end]].transpose()
                    if len(detour_points) >= 3:
                        cur_detour_polygon = Polygon([*detour_points])
                        if cur_detour_polygon.is_valid:
                            detour_polygon = unary_union([detour_polygon, cur_detour_polygon])

                detour_area = detour_polygon.area


        detours_timesteps.append(detour_timesteps)
        detours_duration.append(detour_duration)
        detours_area.append(detour_area)

    trajectories_statistics = DictTree()
    #trajectories_statistics.distance_to_target = distance_to_target
    trajectories_statistics.settling_times = settling_times
    trajectories_statistics.T10s = T10s
    trajectories_statistics.T90s = T90s
    trajectories_statistics.detours_timesteps = detours_timesteps
    trajectories_statistics.detours_duration = jnp.array(detours_duration)
    trajectories_statistics.detours_area = jnp.array(detours_area)

    trajectories_statistics.save(os.path.join(save_folder, "trajectories_statistics.pickle"), overwrite=True)

    return trajectories_statistics


def run_evaluation(config):

    # Pipeline config
    pipeline_config = config.get_pipeline_config()

    # Load history of interventions from the experiment
    print(f"Load Experiment History")
    experiment_history = config.get_experiment_data_history()
    experiment_intervention_params_library = experiment_history.intervention_params_library
    experiment_system_output_library = experiment_history.system_output_library

    # Calc representants statistics
    print(f"Calc representative ids")
    select_representants_config = config.get_select_representants_config()
    if not os.path.exists(os.path.join(pipeline_config.evaluation_data_save_folder, "representants.pickle")):
        representative_ids = select_representants(experiment_system_output_library,
                                                  observed_node_ids=config.observed_node_ids,
                                                  stable_test_n_last_steps=select_representants_config.stable_test_n_last_steps,
                                                  stable_test_settling_threshold=select_representants_config.stable_test_settling_threshold,
                                                  n_representants=select_representants_config.n_representants,
                                                  epsilon=select_representants_config.epsilon,
                                                  n_trials=select_representants_config.n_trials,
                                                  seed=pipeline_config.seed,
                                                  save_folder=pipeline_config.evaluation_data_save_folder)
    else:
        representants = DictTree.load(os.path.join(pipeline_config.evaluation_data_save_folder, "representants.pickle"))
        representative_ids = representants.representative_ids

    print(f"Filter Experiment History")
    experiment_intervention_params_library = jtu.tree_map(lambda node: node[representative_ids],
                                                          experiment_intervention_params_library)
    experiment_system_output_library = jtu.tree_map(lambda node: node[representative_ids],
                                                          experiment_system_output_library)

    # Calc trajectories statistics
    print(f"Calc trajectories statistics")
    if not os.path.exists(os.path.join(pipeline_config.evaluation_data_save_folder, "trajectories_statistics.pickle")):
        trajectories_statistics = calc_trajectories_statistics(experiment_system_output_library.ys[:, jnp.array(config.observed_node_ids), :],
                                                settling_time_threshold=select_representants_config.stable_test_settling_threshold,
                                                save_folder=pipeline_config.evaluation_data_save_folder)
    else:
        trajectories_statistics = DictTree.load(os.path.join(pipeline_config.evaluation_data_save_folder, "trajectories_statistics.pickle"))

    # Create System Modules
    system_rollout_config = config.get_system_rollout_config()
    system_rollout = create_system_rollout_module(system_rollout_config)
    rollout_statistics_encoder_config = config.get_rollout_statistics_encoder_config()
    rollout_statistics_encoder = create_rollout_statistics_encoder_module(rollout_statistics_encoder_config)

    # Create Intervention Modules
    intervention_config = config.get_random_intervention_generator_config()
    _, intervention_fn = create_intervention_module(intervention_config)


    # Run Robustness tests
    T10, T90 = jnp.median(trajectories_statistics.T10s + 1) * system_rollout_config.deltaT, jnp.median(trajectories_statistics.T90s + 1) * system_rollout_config.deltaT
    perturbation_configs = config.get_perturbation_configs(T10, T90)
    for perturbation_config in perturbation_configs:
        # Create Perturbation Module
        perturbation_generator, perturbation_fn = create_perturbation_module(perturbation_config)

        # Run Evaluation Pipeline
        print(f"Run test {perturbation_config.test_task_var_name}_{perturbation_config.test_task_var_val}")
        evaluation_data_save_folder = os.path.join(pipeline_config.evaluation_data_save_folder, f"{perturbation_config.test_task_var_name}_{perturbation_config.test_task_var_val}")
        if not os.path.exists(os.path.join(evaluation_data_save_folder, "history.pickle")):
            run_robustness_tests(pipeline_config.jax_platform_name, pipeline_config.seed,
                                 pipeline_config.n_perturbations, evaluation_data_save_folder,
                                 experiment_system_output_library, experiment_intervention_params_library, intervention_fn,
                                 perturbation_generator, perturbation_fn,
                                 system_rollout, rollout_statistics_encoder,
                                 out_sanity_check=True, save_modules=False, save_logs=True)


if __name__ == "__main__":
    models_obs_database_filepath = "../../resources/bio_models_observed_nodes_database.csv"
    with open(models_obs_database_filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    import argparse
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--rows_ids",
        nargs="*",
        type=int,
        default=list(range(len(rows))),
    )
    args = CLI.parse_args()
    rows = [rows[i] for i in args.rows_ids]

    for row in rows:
        model_idx = int(row["model_idx"])
        model_filepath = f"../../resources/bio_models/model_{model_idx:06d}.py"
        observed_node_ids = list(eval(row["observed_node_ids"]))
        ymin = jnp.array(eval(row["ymin"]))
        ymax = jnp.array(eval(row["ymax"]))
        T = int(row["T"])
        n_nodes = int(row["n_nodes"])

        print(model_idx, observed_node_ids)
        config = evaluation_config.EvaluationConfig(model_idx, model_filepath, observed_node_ids, ymin, ymax, T, n_nodes)
        run_evaluation(config)