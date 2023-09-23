from autodiscjax import DictTree
import jax.numpy as jnp
import jax.tree_util as jtu

if __name__ == "__main__":
    
    examples_save_fp = "evaluation_data_examples.pickle"
    
    experiment_variants = {3: "imgep m=1"}
    

    display_examples = {
    (10, (3,7)): [("noise_std", "0.001", 7, 0, "bottom left"), ("noise_std", "0.005", 25, 0, "bottom right"), 
                  ("noise_std", "0.01", 34, 0, "bottom right"), ("noise_period", "10", 43, 0, "top right"),
                  ("noise_period", "5", 32, 0, "bottom left"),  ("noise_period", "1", 7, 0, "bottom left")],
    (52,(4,7)): [("noise_std", "0.001", 31, 0, "top left"), ("noise_std", "0.005", 36, 0, "top left"), 
                  ("noise_std", "0.01", 27, 0, "top left"), ("noise_period", "10", 19, 0, "top left"),
                  ("noise_period", "5", 35, 0, "top left"),  ("noise_period", "1", 40, 0, "top left")
                ], # (52,(4,7)) or (272,(0,2))
    (647,(2,10)): [("push_magnitude", "0.05", 38, 0, "top right"), ("push_magnitude", "0.1", 6, 0, "top left"), 
                  ("push_magnitude", "0.15", 0, 0, "bottom right"), ("push_number", "1", 10, 0, "bottom right"),
                  ("push_number", "2", 15, 0, "bottom right"),  ("push_number", "3", 13, 0, "bottom right")
                ], # (525,(11,14)) or (647,(2,10))
    (284, (4,6)): [("push_magnitude", "0.05", 4, 0, "top left"), ("push_magnitude", "0.1", 13, 0, "top right"), 
                  ("push_magnitude", "0.15", 30, 0, "bottom right"), ("push_number", "1", 43, 0, "top right"),
                  ("push_number", "2", 32, 0, "top right"),  ("push_number", "3", 3, 0, "top left")
                ], # (284, (4,6))
    (84, (4,6)): [("wall_length", "0.05", 5, 0, "top left"), ("wall_length", "0.1", 7, 0, "top left"), 
                  ("wall_length", "0.15", 15, 0, "top right"), ("wall_number", "1", 35, 0, "top left"),
                  ("wall_number", "2", 21, 0, "top left"),  ("wall_number", "3", 38, 0, "bottom right")
                ], # 38 or 39 for last traj + (84, (4,6)) or 197, (0,3) (should we mix? :))
    (272, (2,3)): [("wall_length", "0.05", 29, 0, "top left"), ("wall_length", "0.1", 28, 0, "bottom right"), 
                  ("wall_length", "0.15", 42, 0, "bottom right"), ("wall_number", "1", 10, 0, "top left"),
                  ("wall_number", "2", 5, 0, "bottom right"),  ("wall_number", "3", 30, 0, "bottom right")
                ], # (272, (2,3))
                   } 
    statistics = DictTree()
    for exp_idx in experiment_variants.keys():
        
        for (system_key, examples) in display_examples.items():
            # Load Expe Data
            model_idx, node_ids = system_key
            root_folder = f"../experiments/experiment_{exp_idx:06d}/model_{model_idx}/nodes_{node_ids[0]}_{node_ids[1]}"
            exp_data_fp = f"{root_folder}/experiment_data/history.pickle"
            exp_data = DictTree.load(exp_data_fp) 
            
            # Load representants and trajectory statistics
            representants_fp = f"{root_folder}/evaluation_data/representants.pickle"
            representants = DictTree.load(representants_fp)
            
            trajectories_statistics_fp = f"{root_folder}/evaluation_data/trajectories_statistics.pickle"
            trajectories_statistics = DictTree.load(trajectories_statistics_fp)

            for (task_name, task_var, eval_traj_idx, perturb_idx, annotation_position) in examples:
                eval_data_fp = f"{root_folder}/evaluation_data/{task_name}_{task_var}/history.pickle"
                eval_data = DictTree.load(eval_data_fp)
                
                exp_traj_idx = representants.representative_ids[eval_traj_idx]
                
                statistics[exp_idx][system_key][task_name][task_var][eval_traj_idx]["system_outputs_prior_perturb"] = jtu.tree_map(lambda node: node[exp_traj_idx], exp_data.system_output_library)
                statistics[exp_idx][system_key][task_name][task_var][eval_traj_idx][perturb_idx]["system_outputs_after_perturb"] = jtu.tree_map(lambda node: node[eval_traj_idx, perturb_idx], eval_data.system_output_library)
                statistics[exp_idx][system_key][task_name][task_var][eval_traj_idx][perturb_idx]["perturbation_params"] = jtu.tree_map(lambda node: node[eval_traj_idx, perturb_idx], eval_data.perturbation_params_library)
                
            # Add details about evaluation config
            deltaT = 0.1
            perturbation_min_duration = float(50)
            perturbation_max_duration = float(500)
            T10, T90 = jnp.median(trajectories_statistics.T10s + 1) * deltaT, jnp.median(trajectories_statistics.T90s + 1) * deltaT
            statistics[exp_idx][system_key]["noise_start"] = T10
            statistics[exp_idx][system_key]["noise_end"] = min(max(T90, T10 + perturbation_min_duration), T10 + perturbation_max_duration)
            for var_val in range(1,4):
                statistics[exp_idx][system_key]["push_ts"][var_val] = jnp.linspace(T10, T90, 2 + var_val)[1:-1]
                
                
    statistics.save(examples_save_fp, overwrite=True)