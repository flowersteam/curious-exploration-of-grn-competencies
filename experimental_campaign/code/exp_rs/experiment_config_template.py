from addict import Dict
from autodiscjax import DictTree
import jax.numpy as jnp
import jax.tree_util as jtu

class ExperimentConfig:

    def __init__(self, model_idx, model_filepath, ymin, ymax, T, n_nodes):
        self.model_idx = model_idx
        self.model_filepath = model_filepath
        self.ymin = ymin
        self.ymax = ymax
        self.T = T
        self.n_nodes = n_nodes

    def get_pipeline_config(self):
        config = Dict()
        config.jax_platform_name = "cpu"
        config.seed = int(<seed>)
        config.n_random_batches = int(<n_random_batches>)
        config.n_imgep_batches = int(<n_imgep_batches>)
        config.batch_size = int(<batch_size>)
        config.experiment_data_save_folder = f"model_{self.model_idx}/experiment_data/"
        return config

    def get_system_rollout_config(self):
        config = Dict()
        config.system_type = "grn"
        config.model_filepath = self.model_filepath
        config.atol = 1e-6
        config.rtol = 1e-12
        config.mxstep = 1000
        config.deltaT = 0.1
        config.n_system_steps = int(self.T/config.deltaT)
        return config

    def get_rollout_statistics_encoder_config(self):
        config = Dict()
        config.statistics_type = "null"
        return config

    def get_random_intervention_generator_config(self):
        config = Dict()
        config.intervention_type = "set_uniform"
        system_rollout_config = self.get_system_rollout_config()
        config.controlled_intervals = [[-system_rollout_config.deltaT/2.0, system_rollout_config.deltaT/2.0]]

        config.controlled_node_ids = list(range(self.n_nodes))
        intervention_params_tree = DictTree()
        for y_idx in config.controlled_node_ids:
            intervention_params_tree.y[y_idx] = "placeholder"

        config.out_treedef = jtu.tree_structure(intervention_params_tree)
        config.out_shape = jtu.tree_map(lambda _: (len(config.controlled_intervals),),
                                        intervention_params_tree)
        config.out_dtype = jtu.tree_map(lambda _: jnp.float32, intervention_params_tree)

        config.low = DictTree()
        config.high = DictTree()
        for y_idx in config.controlled_node_ids:
            config.low.y[y_idx] = self.ymin[y_idx] * jnp.ones(shape=config.out_shape.y[y_idx],
                                                              dtype=config.out_dtype.y[y_idx])
            config.high.y[y_idx] = self.ymax[y_idx] * jnp.ones(shape=config.out_shape.y[y_idx],
                                                               dtype=config.out_dtype.y[y_idx])

        return config

    def get_perturbation_generator_config(self):
        config = Dict()
        config.perturbation_type = "null"
        return config
