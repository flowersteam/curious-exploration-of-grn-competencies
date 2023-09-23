import jax
jax.config.update("jax_platform_name", "cpu")
from autodiscjax.utils.timeseries import is_stable, is_monotonous, is_periodic
import csv
import equinox as eqx
from functools import partial
import itertools
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import sbmltoodejax
from shapely import Point, unary_union
import time
import matplotlib.pyplot as plt

class SimpleModelStep(eqx.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @jit
    def __call__(self, y, w, c, t, deltaT):
        n = len(y)
        W = c[:n * n].reshape((n, n))
        B = c[n * n:(n + 1) * n]
        Tau = c[(n + 1) * n:(n + 2) * n]
        y_new = deltaT / Tau * jax.nn.sigmoid(W @ y + B) + (1 - deltaT / Tau) * y
        t_new = t + deltaT
        w_new = w

        return y_new, w_new, c, t_new

class ModelRollout(eqx.Module):
    deltaT: float = 0.1
    modelstepfunc: SimpleModelStep

    def __init__(self, deltaT=0.1, modelstepfunc=SimpleModelStep(), **kwargs):
        super().__init__()
        self.deltaT = deltaT
        self.modelstepfunc = modelstepfunc

    @partial(jit, static_argnames=("n_steps",))
    def __call__(self, n_steps, y0, w0, c, t0):

        def f(carry, x):
            y, w, c, t = carry
            return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)

        (y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
        ys = jnp.moveaxis(ys, 0, -1)
        ws = jnp.moveaxis(ws, 0, -1)
        return ys, ws, ts


if __name__ == '__main__':

    key = jrandom.PRNGKey(0)

    # Hyperparameters
    t0 = 0.0
    deltaT = 0.1
    atol = 1e-6
    rtol = 1e-12
    mxstep = 1000
    n_secs = 250 # 250 instead of 2500 as biological networks converge much faster
    n_system_steps = int(n_secs / deltaT)

    y0_min = 0.
    y0_max = 1.
    W_min = -30.
    W_max = 30.
    B_min = -10.
    B_max = 10.
    T_min = 1
    T_max = 15

    batch_size = 50
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
    epsilon = 0.05

    in_csv_filepath = "bio_models_database.csv"
    with open(in_csv_filepath, 'r') as f:
        reader = csv.DictReader(f)

        all_n_nodes = []
        all_in_degrees = []

        for row_idx, in_row in enumerate(reader):
            n_nodes = int(in_row["n_nodes"])
            all_n_nodes.append(n_nodes)
            in_degrees = eval(in_row["n_inputs_per_node"])
            all_in_degrees += in_degrees


    all_n_nodes = jnp.array(all_n_nodes)
    all_in_degrees = jnp.array(all_in_degrees)

    n_nodes_mean = all_n_nodes.mean()
    n_nodes_std = all_n_nodes.std()
    n_nodes_min = all_n_nodes.min()
    n_nodes_max = all_n_nodes.max()
    print(f"Number of nodes: {n_nodes_mean:.2f} (mean), {n_nodes_std:.2f} (std), {n_nodes_min:.2f} (min), {n_nodes_max:.2f} (max)")
    in_degrees_mean = all_in_degrees.mean()
    in_degrees_std = all_in_degrees.std()
    in_degrees_min = all_in_degrees.min()
    in_degrees_max = all_in_degrees.max()
    print(f"Nodes in-degree: {in_degrees_mean:.2f} (mean), {in_degrees_std:.2f} (std), {in_degrees_min:.2f} (min), {in_degrees_max:.2f} (max)")

    n_bio_models = len(all_n_nodes)
    n_random_models = 10 * n_bio_models

    key = jrandom.PRNGKey(0)
    n_total_systems = 0
    valid_systems_versatility = []
    valid_systems_n_covered_bins = []
    for model_idx in range(1, n_random_models+1):
        # sample n_nodes, in_degrees and y0
        key, subkey = jrandom.split(key)
        n_nodes = min(max(int(jnp.round(n_nodes_mean + n_nodes_std * jrandom.normal(subkey, shape=()))), n_nodes_min), n_nodes_max)
        key, subkey = jrandom.split(key)
        in_degrees = jnp.minimum(jnp.maximum(jnp.round(in_degrees_mean + in_degrees_std * jrandom.normal(subkey, shape=(n_nodes, ))).astype(jnp.int32), in_degrees_min), in_degrees_max)

        # Sample y0, c
        key, subkey = jrandom.split(key)
        y0 = jrandom.uniform(subkey, shape=(n_nodes,), minval=y0_min, maxval=y0_max)

        key, subkey = jrandom.split(key)
        W = jrandom.uniform(subkey, shape=(n_nodes, n_nodes), minval=W_min, maxval=W_max)
        for node_idx in range(n_nodes):
            n_clamped_inputs = max(n_nodes-in_degrees[node_idx], 0)
            key, subkey = jrandom.split(key)
            clamped_input_ids = jrandom.choice(subkey, jnp.arange(n_nodes), shape=(n_clamped_inputs, ), replace=False)
            W = W.at[row_idx, clamped_input_ids].set(0.0)
        key, subkey = jrandom.split(key)
        B = jrandom.uniform(subkey, shape=(n_nodes,), minval=B_min, maxval=B_max)
        key, subkey = jrandom.split(key)
        T = jrandom.uniform(subkey, shape=(n_nodes,), minval=T_min, maxval=T_max)
        c = jnp.concatenate([W.flatten(), B, T])

        # Generate model
        grn_step = SimpleModelStep()
        model = ModelRollout(deltaT=deltaT, modelstepfunc=grn_step)

        ## Simulate in batch mode with random inits between ymin and ymax
        key, subkey = jrandom.split(key)
        y0 = jrandom.uniform(subkey, shape=(batch_size, n_nodes), minval=y0_min, maxval=y0_max)
        w0 = jnp.tile(jnp.array([]), (batch_size, 1))
        model = vmap(model, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, None))
        ys, ws, ts = model(n_system_steps, y0, w0, c, t0)

        pair_combinations = list(itertools.combinations(range(n_nodes), 2))
        for observed_node_ids in pair_combinations:

            # Check NaN values
            observed_ys = ys[:, jnp.array(observed_node_ids), :]
            valid_trajectories_ids = ~(jnp.isnan(observed_ys).any(-1).any(-1))
            valid_trajectories_ratio = valid_trajectories_ids.sum() / batch_size
            valid_observed_ys = observed_ys[valid_trajectories_ids]

            if valid_trajectories_ratio > 0:
                # Check valid trajectories values
                valid_observed_ys = ys[valid_trajectories_ids][:, jnp.array(observed_node_ids), :]
                n_total_systems += 1

                # Check convergence / periodicity
                is_stable_bool, _, _ = is_stable(valid_observed_ys,
                                                 time_window=jnp.r_[-stable_test_n_last_steps:0],
                                                 settling_threshold=stable_test_settling_threshold)
                stable_trajectories_ratio = is_stable_bool.sum() / is_stable_bool.size

                is_monotonous_bool, diff_signs = is_monotonous(valid_observed_ys,
                                                               time_window=jnp.r_[-periodic_test_n_last_steps:0])
                is_periodic_bool, _, _, _ = is_periodic(valid_observed_ys,
                                                        time_window=jnp.r_[-periodic_test_n_last_steps:0],
                                                        deltaT=deltaT,
                                                        max_frequency_threshold=periodic_test_max_frequency_threshold)
                is_periodic_bool = is_periodic_bool & (~is_stable_bool) & (~is_monotonous_bool)
                periodic_trajectories_ratio = is_periodic_bool.sum() / is_periodic_bool.size

                if (valid_trajectories_ratio >= valid_trajectories_min_ratio) and \
                        (stable_trajectories_ratio >= stable_trajectories_min_ratio) and \
                        (periodic_trajectories_ratio <= periodic_trajectories_max_ratio):

                    # Check goal space extent
                    reached_goal_points = valid_observed_ys[is_stable_bool.all(-1), :, -1]
                    reached_space_extent = jnp.nanmax(reached_goal_points, 0) - jnp.nanmin(reached_goal_points, 0)

                    if (reached_space_extent >= goal_space_extend_min).all():

                        # CALC VERSATILITY as in Fig 7
                        # normalize
                        reached_endpoints = reached_goal_points/reached_space_extent

                        # calc versatility
                        covered_area = unary_union([Point(endpoint).buffer(epsilon) for endpoint in reached_endpoints]).area
                        versatility = covered_area / (1 + 2 * epsilon) ** 2 # normalize between [0,1]
                        valid_systems_versatility.append(versatility)

                        if versatility > 0.026:
                            fig, ax = plt.subplots(1,3)
                            for traj in valid_observed_ys[is_stable_bool.all(-1)]:
                                ax[0].plot(traj[0, -1000:])
                                ax[1].plot(traj[1, -1000:])
                            ax[2].scatter(reached_goal_points[:, 0], reached_goal_points[:, 1])
                            plt.show()


    valid_systems_versatility = jnp.array(valid_systems_versatility)
    ordered_ids = valid_systems_versatility.argsort()
    selected_systems_versatility = valid_systems_versatility[ordered_ids[-432:]]
    print(f"There are {len(valid_systems_versatility)} systems passing F1, F2, F3 filters over the total of {n_total_systems} random systems")
    print(f"Statistics of the 432 more versatile systems:")
    print(f"\t Versatility: {selected_systems_versatility.mean():.3f} (mean), {selected_systems_versatility.std():.3f} (std), "
          f"{selected_systems_versatility.min():.3f} (min), {selected_systems_versatility.max():.3f} (max)")

    jnp.save("random_networks_versatility.npy", selected_systems_versatility)