import os.path

from autodiscjax.utils.timeseries import is_stable, is_monotonous, is_periodic
import csv
import importlib
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jrandom
import itertools
import os
import random
import shutil
import time

jax.config.update("jax_platform_name", "cpu")


if __name__ == '__main__':
    key = jrandom.PRNGKey(0)

    # Hyperparameters
    t0 = 0.0
    deltaT = 0.1
    atol = 1e-6
    rtol = 1e-12
    mxstep = 1000
    n_secs = 2500
    n_system_steps = int(n_secs / deltaT)

    rmin = 0.05
    rmax = 20
    batch_size = 50

    n_nodes_min = 3
    simu_time_default_max = 1.0
    simu_time_batch_max = 15.0
    valid_trajectories_min_ratio = 0.8
    periodic_trajectories_max_ratio = 0.2
    stable_trajectories_min_ratio = 0.8
    periodic_test_n_last_steps = n_system_steps // 2
    periodic_test_max_frequency_threshold = 40
    stable_test_n_last_steps = 1000
    stable_test_settling_threshold = 0.02
    n_bins = 20
    goal_space_extend_min = 1e-1
    n_covered_bins_min = 5

    # I/O files
    models_csv_filepath = "bio_models_preselection_statistics.csv"
    observed_nodes_csv_filepath = "bio_models_observed_nodes_selection_statistics.csv"
    observed_nodes_fieldnames = ["model_idx", "observed_node_ids", "batch_size", "T", "ymin", "ymax",
                                 f'simu_time_jz (b={batch_size}, T={n_secs})',
                                 "valid_trajectories_ratio", "stable_trajectories_ratio", "periodic_trajectories_ratio",
                                 "reached_space_extent", "n_covered_bins"]
    observed_nodes_database_filepath = "bio_models_observed_nodes_database.csv"
    models_database_filepath = "bio_models_database.csv"

    if os.path.exists(observed_nodes_csv_filepath) or \
            os.path.exists(observed_nodes_database_filepath) or \
            os.path.exists(models_database_filepath):
        raise FileExistsError


    with open(models_csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)


    with open(observed_nodes_csv_filepath, "w") as f:
        writer = csv.DictWriter(f, observed_nodes_fieldnames)
        writer.writeheader()

    valid_model_nodes_pair = []
    for row in rows:
        model_idx = int(row["model_idx"])
        print(model_idx)

        if (row["simu_time (T=10)"] == "") or (float(row["simu_time (T=10)"]) > simu_time_default_max) or (int(row["n_nodes"]) < n_nodes_min):
            pass

        else:
            n_nodes = int(row["n_nodes"])

            ## Load Model
            jax_filepath = f"jax_files/BIOMD{model_idx:010d}.py"
            spec = importlib.util.spec_from_file_location("ModelSpec", jax_filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_cls = getattr(module, "ModelRollout")
            model = model_cls(deltaT=deltaT, atol=atol, rtol=rtol, mxstep=mxstep)
            w0 = getattr(module, "w0")
            c = getattr(module, "c")

            ## Simulate default rollout
            ys_default, _, _ = model(n_system_steps)
            ys_default.block_until_ready()

            ## Simulate in batch mode with random inits between ymin and ymax
            ymin = rmin * jnp.nanmin(ys_default, -1)
            ymin_valid = (ymin>=-1e-6).all()
            ymax = rmax * jnp.nanmax(ys_default, -1)
            key, subkey = jrandom.split(key)
            y0 = jrandom.uniform(subkey, shape=(batch_size, n_nodes), minval=ymin, maxval=ymax)
            w0 = jnp.tile(w0, (batch_size, 1))
            model = vmap(model, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, None))

            simu_start = time.time()
            ys, ws, ts = model(n_system_steps, y0, w0, c, t0)
            ys.block_until_ready()
            simu_end = time.time()

            ## Add statistics from batch rollouts
            ### simu time
            simu_time = simu_end - simu_start

            ### valid trajectories ratio
            valid_trajectories_ids = ~(jnp.isnan(ys).any(-1).any(-1)) & (ys>=-1e-6).all(-1).all(-1)
            valid_trajectories_ratio = valid_trajectories_ids.sum() / batch_size

            T = n_secs
            pair_combinations = list(itertools.combinations(range(n_nodes), 2))
            for observed_node_ids in pair_combinations:
                print(observed_node_ids)

                if valid_trajectories_ratio > 0:
                    # Check valid trajectories values
                    valid_observed_ys = ys[valid_trajectories_ids][:, jnp.array(observed_node_ids), :]

                    # Check goal space extent / number of covered bins
                    reached_goal_points = valid_observed_ys[:, :, -1]
                    x = reached_goal_points[:, 0]
                    y = reached_goal_points[:, 1]
                    Hf, xedges, yedges = jnp.histogram2d(x=x, y=y, bins=n_bins, range=[[jnp.nanmin(x), jnp.nanmax(x)],
                                                                                       [jnp.nanmin(y), jnp.nanmax(y)]])
                    reached_space_extent = jnp.nanmax(reached_goal_points, 0) - jnp.nanmin(reached_goal_points, 0)
                    n_covered_bins = (Hf > 0).sum()

                    # Check convergence / periodicity
                    is_stable_bool, _, _ = is_stable(valid_observed_ys,
                                                     time_window=jnp.r_[-stable_test_n_last_steps:0],
                                                     settling_threshold=stable_test_settling_threshold)
                    stable_trajectories_ratio = is_stable_bool.sum() / is_stable_bool.size

                    is_monotonous_bool, diff_signs = is_monotonous(valid_observed_ys, time_window=jnp.r_[-periodic_test_n_last_steps:0])
                    is_periodic_bool, _, _, _ = is_periodic(valid_observed_ys,
                                                            time_window=jnp.r_[-periodic_test_n_last_steps:0],
                                                            deltaT=deltaT,
                                                            max_frequency_threshold=periodic_test_max_frequency_threshold)
                    is_periodic_bool = is_periodic_bool & (~is_stable_bool) & (~is_monotonous_bool)
                    periodic_trajectories_ratio = is_periodic_bool.sum() / is_periodic_bool.size

                else:
                    stable_trajectories_ratio = 0.0
                    periodic_trajectories_ratio = 0.0
                    reached_space_extent = jnp.zeros(len(observed_node_ids))
                    n_covered_bins = 0


                ## Add statistics
                ymin = [float(y) for y in ymin]
                ymax = [float(y) for y in ymax]
                with open(observed_nodes_csv_filepath, "a") as f:
                    writer = csv.DictWriter(f, observed_nodes_fieldnames)
                    observed_nodes_row = {"model_idx": model_idx, "observed_node_ids": observed_node_ids,
                                          "batch_size": batch_size, "T": T, "ymin": ymin, "ymax": ymax,
                                          f"simu_time_jz (b={batch_size}, T={n_secs})": simu_time,
                                          "valid_trajectories_ratio": valid_trajectories_ratio,
                                          "stable_trajectories_ratio": stable_trajectories_ratio,
                                          "periodic_trajectories_ratio": periodic_trajectories_ratio,
                                          "reached_space_extent": reached_space_extent, "n_covered_bins": n_covered_bins}
                    writer.writerow(observed_nodes_row)

                if ymin_valid and (simu_time <= simu_time_batch_max) and\
                        (valid_trajectories_ratio >= valid_trajectories_min_ratio) and \
                        (stable_trajectories_ratio >= stable_trajectories_min_ratio) and \
                        (periodic_trajectories_ratio <= periodic_trajectories_max_ratio) and \
                        (reached_space_extent >= goal_space_extend_min).all() and \
                        (n_covered_bins >= n_covered_bins_min):
                    valid_model_nodes_pair.append((model_idx, observed_node_ids, n_nodes, row["n_inputs_per_node"], ymin, ymax, T))


    random.seed(0)
    random.shuffle(valid_model_nodes_pair)
    with open(observed_nodes_database_filepath, "w") as f:
        writer = csv.DictWriter(f, ["model_idx", "observed_node_ids", "n_nodes", "n_inputs_per_node", "ymin", "ymax", "T"])
        writer.writeheader()
        for (model_idx, observed_node_ids, n_nodes, n_inputs_per_node, ymin, ymax, T) in valid_model_nodes_pair:
            writer.writerow({"model_idx": model_idx, "observed_node_ids": observed_node_ids,
                             "n_nodes": n_nodes, "n_inputs_per_node": n_inputs_per_node,
                             "ymin": ymin, "ymax": ymax, "T": T})

    ordered_models = []
    ordered_models_infos = []
    with open(observed_nodes_database_filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_idx = int(row["model_idx"])
            if model_idx not in ordered_models:
                ordered_models.append(model_idx)
                ordered_models_infos.append((model_idx, int(row["n_nodes"]), eval(row["n_inputs_per_node"]),
                                             eval(row["ymin"]), eval(row["ymax"]), int(row["T"])))


    with open(models_database_filepath, "w") as f:
        writer = csv.DictWriter(f, ["model_idx", "n_nodes", "n_inputs_per_node", "ymin", "ymax", "T"])
        writer.writeheader()
        for (model_idx, n_nodes, n_inputs_per_node, ymin, ymax, T) in ordered_models_infos:
            writer.writerow({"model_idx": model_idx, "n_nodes": n_nodes, "n_inputs_per_node": n_inputs_per_node, "ymin": ymin, "ymax": ymax, "T": T})

            # Copy jax files
            shutil.copy(f"./jax_files/BIOMD{model_idx:010d}.py", f"./bio_models/model_{model_idx:06d}.py")



