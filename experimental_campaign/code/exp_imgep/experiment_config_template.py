from addict import Dict
from autodiscjax import DictTree
import jax.numpy as jnp
import jax.tree_util as jtu

class ExperimentConfig:

    def __init__(self, model_idx, model_filepath, observed_node_ids, ymin, ymax, T, n_nodes):
        self.model_idx = model_idx
        self.model_filepath = model_filepath
        self.observed_node_ids = observed_node_ids
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
        assert len(self.observed_node_ids) == 2
        config.experiment_data_save_folder = f"model_{self.model_idx}/nodes_{self.observed_node_ids[0]}_{self.observed_node_ids[1]}/experiment_data/"
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

    def get_goal_embedding_encoder_config(self):
        config = Dict()
        config.encoder_type = "filter"
        goal_embedding_tree = "placeholder"
        config.out_treedef = jtu.tree_structure(goal_embedding_tree)
        config.out_shape = jtu.tree_map(lambda _: (len(self.observed_node_ids),), goal_embedding_tree)
        config.out_dtype = jtu.tree_map(lambda _: jnp.float32, goal_embedding_tree)
        config.filter_fn = jtu.Partial(lambda system_outputs: system_outputs.ys[..., self.observed_node_ids, -1])

        return config

    def get_goal_generator_config(self):
        config = Dict()

        goal_embedding_encoder_config = self.get_goal_embedding_encoder_config()
        config.out_treedef = goal_embedding_encoder_config.out_treedef
        config.out_shape = goal_embedding_encoder_config.out_shape
        config.out_dtype = goal_embedding_encoder_config.out_dtype

        config.low = 0.0
        config.high = None
        config.generator_type = "<generator_type>"

        if config.generator_type == "hypercube":
            config.hypercube_scaling = float(<hypercube_scaling>)

        elif config.generator_type == "IMFlow":
            optimizer_config = self.get_gc_intervention_optimizer_config()
            pipeline_config = self.get_pipeline_config()
            intervention_selector_config = self.get_gc_intervention_selector_config()
            config.distance_fn = intervention_selector_config.loss_f
            config.IM_val_scaling = float(<IM_val_scaling>)
            config.IM_grad_scaling = float(<IM_grad_scaling>)
            config.random_proba = float(<IM_random_proba>)
            config.flow_noise = float(<IM_flow_noise>)
            config.time_window = jnp.r_[-pipeline_config.batch_size*optimizer_config.n_optim_steps*optimizer_config.n_workers:0]

        else:
            raise ValueError

        return config

    def get_goal_achievement_loss_config(self):
        config = Dict()
        config.loss_type = "L2"
        return config

    def get_gc_intervention_selector_config(self):
        config = Dict()
        config.selector_type = "nearest_neighbor"
        config.loss_f = jtu.Partial(lambda y, x: jnp.sqrt(jnp.square(y - x).sum(-1)))
        config.k = int(<k>)
        return config

    def get_gc_intervention_optimizer_config(self):
        config = Dict()

        random_intervention_generator_config = self.get_random_intervention_generator_config()
        config.out_treedef = random_intervention_generator_config.out_treedef
        config.out_shape = random_intervention_generator_config.out_shape
        config.out_dtype = random_intervention_generator_config.out_dtype

        config.low = random_intervention_generator_config.low
        config.high = random_intervention_generator_config.high

        config.optimizer_type = "<optimizer_type>"
        config.n_optim_steps = int(<n_optim_steps>)
        config.n_workers = int(<n_workers>)
        config.init_noise_std = jtu.tree_map(lambda low, high: float(<init_noise_std>) * (high - low), config.low, config.high)

        if config.optimizer_type == "SGD":
            config.lr = jtu.tree_map(lambda low, high: float(<lr>) * (high - low), config.low, config.high)

        return config
