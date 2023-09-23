from autodiscjax import DictTree
from autodiscjax.utils.timeseries import is_stable
import csv
import jax.numpy as jnp
import os
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

def calc_coverage(reached_endpoints, epsilon):

    for step_idx, reached_endpoint in enumerate(reached_endpoints):
        if step_idx == 0:
            union_polygon = Point(reached_endpoints[0]).buffer(epsilon)
            covered_areas = [union_polygon.area]
        else:
            union_polygon = unary_union([union_polygon, Point(reached_endpoint).buffer(epsilon)])
            covered_areas.append(union_polygon.area)

    return union_polygon, covered_areas

def fill_from_left(a, x=0):
    to_fill = (a == x)
    lefter_not_x = jnp.where(~to_fill)[0][0]
    a_start = a[:lefter_not_x]
    a = a[lefter_not_x:]
    to_fill = to_fill[lefter_not_x:]
    if to_fill.any():
        lefts = ~to_fill & (jnp.roll(a, -1) == x)
        fill_from = lefts.cumsum()
        fill_with = a[jnp.where(lefts)[0]][fill_from - 1]
        a = a.at[to_fill].set(fill_with[to_fill])
    a = jnp.concatenate([a_start, a])
    return a

if __name__ == "__main__":

    statistics_save_fp = "experiment_data_statistics.pickle"
    if os.path.exists(statistics_save_fp):
        statistics = DictTree.load(statistics_save_fp)
    else:
        statistics = DictTree()

    experiment_variants = {1: "random search", 2: "imgep m=3", 3: "imgep m=1"}
    epsilons = [0.033, 0.05, 0.1]

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
        if isinstance(statistics[system_key].union_polygon[0][epsilons[-1]], Polygon):
            continue

        print(system_key)
        model_idx = system_key[0]
        observed_nodes_ids = system_key[1]

        all_reached_endpoints = DictTree()
        for exp_idx, exp_name in experiment_variants.items():
            # load experiment history
            exp_folder = f"../experiments/experiment_{exp_idx:06d}/model_{model_idx}"
            if "imgep" in exp_name:
                exp_folder += f"/nodes_{observed_nodes_ids[0]}_{observed_nodes_ids[1]}"
            exp_history_fp = f"{exp_folder}/experiment_data/history.pickle"
            exp_history = DictTree.load(exp_history_fp)

            # filter non-valid and non-stable trajectories
            trajectories = exp_history.system_output_library.ys[:, observed_nodes_ids, :]
            is_valid_bool = ~(jnp.isnan(trajectories.any(-1).any(-1))) & ((trajectories>=-1e-6).all(-1).all(-1))
            statistics[system_key].is_valid_bool[exp_idx] = is_valid_bool

            is_stable_bool, _, _ = is_stable(trajectories, time_window=jnp.r_[-1000:0], settling_threshold=0.02)
            is_stable_bool = is_stable_bool.all(-1)
            if is_stable_bool.sum() / is_stable_bool.size < 1.0:
                print(f"{exp_name} - is_stable_ratio: {is_stable_bool.sum() / is_stable_bool.size}")
            statistics[system_key].is_stable_bool[exp_idx] = is_stable_bool

            # get reached endpoints
            reached_endpoints = trajectories[is_valid_bool & is_stable_bool, : , -1]
            statistics[system_key].reached_endpoints[exp_idx] = reached_endpoints

        # calc BC space extent
        all_reached_endpoints_array = jnp.concatenate([statistics[system_key].reached_endpoints[exp_idx] for exp_idx in experiment_variants.keys()], axis=0)
        analytic_bc_space_min = jnp.nanmin(all_reached_endpoints_array, axis=0)
        statistics[system_key].analytic_bc_space_min = analytic_bc_space_min
        analytic_bc_space_max = jnp.nanmax(all_reached_endpoints_array, axis=0)
        statistics[system_key].analytic_bc_space_max = analytic_bc_space_max

        # normalize endpoints
        for exp_idx, reached_endpoints in statistics[system_key].reached_endpoints.items():
            statistics[system_key].normalized_reached_endpoints[exp_idx] = (reached_endpoints-analytic_bc_space_min)/(analytic_bc_space_max-analytic_bc_space_min)

        # calc union polygon and covered areas
        for epsilon in epsilons:
            for exp_idx, exp_name in experiment_variants.items():
                union_polygon, covered_areas = calc_coverage(statistics[system_key].normalized_reached_endpoints[exp_idx], epsilon)
                # fill covered areas with zeros for non-valid trajectories
                is_valid_bool = statistics[system_key].is_valid_bool[exp_idx] & statistics[system_key].is_stable_bool[exp_idx]
                filled_covered_areas = jnp.zeros(shape=(len(is_valid_bool), ), dtype=jnp.float32)
                filled_covered_areas = filled_covered_areas.at[is_valid_bool].set(jnp.array(covered_areas))
                statistics[system_key].covered_areas[exp_idx][epsilon] = fill_from_left(filled_covered_areas, x=0)
                statistics[system_key].union_polygon[exp_idx][epsilon] = union_polygon

        n_new_keys += 1
        if n_new_keys % 5 == 0:
            statistics.save(statistics_save_fp, overwrite=True)

    statistics.save(statistics_save_fp, overwrite=True)
