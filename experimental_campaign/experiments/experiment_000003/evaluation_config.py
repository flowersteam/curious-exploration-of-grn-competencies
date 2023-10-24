from addict import Dict
from autodiscjax import DictTree
from experiment_config import ExperimentConfig
import jax.numpy as jnp
import os

class EvaluationConfig (ExperimentConfig):

    def get_select_representants_config(self):
        config = Dict()
        config.stable_test_n_last_steps = 1000
        config.stable_test_settling_threshold = 0.02
        config.n_representants = int(45)
        config.epsilon = 0.05
        config.n_trials = 500
        return config

    def get_experiment_data_history(self):
        return DictTree.load(os.path.join(self.get_pipeline_config().experiment_data_save_folder, "history.pickle"))

    def get_pipeline_config(self):
        config = super().get_pipeline_config()
        config.n_perturbations = int(3)
        config.evaluation_data_save_folder =  f"model_{self.model_idx}/nodes_{self.observed_node_ids[0]}_{self.observed_node_ids[1]}/evaluation_data/"
        return config

    def get_system_rollout_config(self):
        config = super().get_system_rollout_config()
        config.n_system_steps = int(config.n_system_steps*1.2)
        return config

    def get_perturbation_configs(self, T10, T90):
        configs = []

        test_tasks = {
            "noise_std": [0.001, 0.005, 0.01],
            "noise_period": [10, 5, 1],
            "push_magnitude": [0.05, 0.1, 0.15],
            "push_number": [1, 2, 3],
            "wall_length": [0.05, 0.1, 0.15],
            "wall_number": [1, 2, 3]
        }
        
        deltaT = self.get_system_rollout_config().deltaT
        n_secs = self.get_system_rollout_config().n_system_steps * deltaT
        
        perturbation_min_duration = float(50)
        perturbation_max_duration = float(500)

        for var_name, test_task_var_range in test_tasks.items():
            for var_val in test_task_var_range:
                config = Dict()
                config.test_task_var_name = var_name
                config.test_task_var_val = var_val

                if var_name.split("_")[0] == "noise":
                    start = T10
                    end = min(max(T90, start + perturbation_min_duration), start + perturbation_max_duration)
                    config.perturbed_node_ids = list(range(self.n_nodes))
                    config.perturbation_type = "noise"

                    if var_name.split("_")[1] == "std":
                        n_noises = int((end-start)//float(5))
                        config.perturbed_intervals = [[t-deltaT/2, t+deltaT/2] for t in jnp.linspace(start, end, 2 + n_noises)[1:-1]] #5
                        config.std = var_val

                    elif var_name.split("_")[1] == "period":
                        n_noises = int((end - start) // float(var_val))
                        config.perturbed_intervals = [[t-deltaT/2, t+deltaT/2] for t in jnp.linspace(start, end, 2 + n_noises)[1:-1]]
                        config.std = float(0.005)

                elif var_name.split("_")[0] == "push":
                    config.perturbed_node_ids = self.observed_node_ids
                    config.perturbation_type = "push"

                    if var_name.split("_")[1] == "magnitude":
                        config.perturbed_intervals = [[(T90+T10)/2.-deltaT/2, (T90+T10)/2.+deltaT/2]]
                        config.magnitude = var_val

                    elif var_name.split("_")[1] == "number":
                        config.perturbed_intervals = [[t-deltaT/2, t+deltaT/2] for t in jnp.linspace(T10, T90, 2 + var_val)[1:-1]]
                        config.magnitude = float(0.1)

                elif var_name.split("_")[0] == "wall":
                    config.perturbed_node_ids = self.observed_node_ids
                    config.perturbation_type = "wall"
                    config.wall_type = "force_field"
                    config.perturbed_intervals = [[deltaT*2, n_secs]] #/!\ starting at zero will lead to wrong init states, must start >= deltaT
                    config.walls_sigma = [1e-2, 1e-4]

                    if var_name.split("_")[1] == "length":
                        config.n_walls = 1
                        config.walls_intersection_window = [[0.1, 0.9]]
                        config.walls_length_range = [[var_val, var_val]]

                    elif var_name.split("_")[1] == "number":
                        config.n_walls = var_val
                        walls_windows_pos = jnp.linspace(0, 1, 2 + var_val)
                        walls_spacing = (walls_windows_pos[1] - walls_windows_pos[0]) * 1/4
                        config.walls_intersection_window = [[t - walls_spacing, t + walls_spacing] for t in walls_windows_pos[1:-1]]
                        config.walls_length_range = [[0.1, 0.1]] * var_val

                configs.append(config)

        return configs