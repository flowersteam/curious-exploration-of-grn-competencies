import os.path

from autodiscjax import DictTree
import csv
import jax.numpy as jnp

if __name__ == "__main__":
    statistics_save_fp = "evaluation_data_statistics.pickle"
    if os.path.exists(statistics_save_fp):
        statistics = DictTree.load(statistics_save_fp)
    else:
        statistics = DictTree()

    experiment_variants = {3: "imgep m=1"}
    test_tasks = {
        "noise_std": [0.001, 0.005, 0.01],
        "noise_period": [10, 5, 1],
        "push_magnitude": [0.05, 0.1, 0.15],
        "push_number": [1, 2, 3],
        "wall_length": [0.05, 0.1, 0.15],
        "wall_number": [1, 2, 3]
    }
    deltaT = 0.1


    models_observed_nodes_database_filepath = "../resources/bio_models_observed_nodes_database.csv"
    system_keys = []
    n_new_keys = 0
    with open(models_observed_nodes_database_filepath, "r") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            model_idx = int(row["model_idx"])
            observed_nodes_ids = eval(row["observed_node_ids"])
            system_key = (model_idx, observed_nodes_ids)
            system_keys.append(system_key)

    for key_idx, system_key in enumerate(system_keys):

        # check if last entry has been added, it yes skip system key
        last_exp_idx = list(experiment_variants.keys())[-1]
        last_test_task_name = list(test_tasks.keys())[-1]
        last_test_task_val = test_tasks[last_test_task_name][-1]
        if isinstance(statistics[system_key].endpoints_sensitivity[last_exp_idx][last_test_task_name][last_test_task_val], jnp.ndarray):
            continue

        print(system_key)
        model_idx = system_key[0]
        observed_nodes_ids = system_key[1]

        for exp_idx, exp_name in experiment_variants.items():

            exp_folder = f"../experiments/experiment_{exp_idx:06d}/model_{model_idx}"
            if "imgep" in exp_name:
                exp_folder += f"/nodes_{observed_nodes_ids[0]}_{observed_nodes_ids[1]}"

            if not os.path.exists(f"{exp_folder}/evaluation_data/representants.pickle"):
                continue

            # load representants statistics
            representants = DictTree.load(f"{exp_folder}/evaluation_data/representants.pickle")
            statistics[system_key].representants_ids[exp_idx] = representants.representative_ids
            statistics[system_key].representants_covered_ratio[exp_idx] = representants.covered_ratio

            # load trajectories statistics
            """
            trajectories_statistics = DictTree.load(f"{exp_folder}/evaluation_data/trajectories_statistics.pickle")
            for k, v in trajectories_statistics.items():
                if k == "detours_timesteps":
                    continue
                if len(trajectories_statistics[k]) > 0:
                    if k in ['settling_times', 'T10s', 'T90s', 'detours_duration']:
                        if k in ['settling_times', 'T10s', 'T90s']:
                            v = v+1
                        v = v*deltaT
                    stat_mean, stat_std, stat_med, stat_P25, stat_P75, stat_min, stat_max = (jnp.nanmean(v), jnp.nanstd(v),
                                                                                             jnp.median(v), jnp.nanpercentile(v, 25), jnp.nanpercentile(v, 75),
                                                                                             jnp.nanmin(v), jnp.nanmax(v))
                    statistics[system_key][k][exp_idx] = [stat_mean, stat_std, stat_med, stat_P25, stat_P75, stat_min, stat_max]
                else:
                    statistics[system_key][k][exp_idx] = None
            """


            # load robustness statistics
            ## load trajectories prior perturbation
            experiment_history = DictTree.load(f"{exp_folder}/experiment_data/history.pickle")
            representants_trajectories = experiment_history.system_output_library.ys[representants.representative_ids]

            trajectories_prior_perturbation = representants_trajectories[:, jnp.array(observed_nodes_ids), 1:] # remove first step (dont wanna take into account big jumps happening in first step)
            trajectories_extent = (trajectories_prior_perturbation.max(-1) - trajectories_prior_perturbation.min(-1))
            #trajectories_extent = trajectories_extent.at[trajectories_extent == 0.].set(1.) #here we to discard sensitivity metric when trajectories_extent=0
            X = trajectories_prior_perturbation[:, :, -1]
            statistics[system_key].endpoints_prior_perturbation[exp_idx] = X

            for test_task_var_name, test_task_var_range in test_tasks.items():
                for test_task_var_val in test_task_var_range:
                    test_folder = f"{exp_folder}/evaluation_data/{test_task_var_name}_{test_task_var_val}"
                    test_experiment_history = DictTree.load(f"{test_folder}/history.pickle")
                    perturbed_X = test_experiment_history.system_output_library.ys[:, :, jnp.array(observed_nodes_ids),-1]
                    sensitivity = jnp.sqrt((((X[:, jnp.newaxis, :] - perturbed_X) / trajectories_extent[:, jnp.newaxis, :]) ** 2).sum(-1)).mean(-1)
                    statistics[system_key].endpoints_after_perturbation[exp_idx][test_task_var_name][test_task_var_val] = perturbed_X
                    statistics[system_key].endpoints_sensitivity[exp_idx][test_task_var_name][test_task_var_val] = sensitivity

        n_new_keys += 1
        if n_new_keys % 5 == 0:
            statistics.save(statistics_save_fp, overwrite=True)

    statistics.save(statistics_save_fp, overwrite=True)
