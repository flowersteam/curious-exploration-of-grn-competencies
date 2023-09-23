from autodiscjax.experiment_pipelines import run_imgep_experiment
from autodiscjax.utils.create_modules import *
import csv
import experiment_config
import jax.numpy as jnp
import os
import sys

def run_experiment(config):
    # Get Pipeline Config
    pipeline_config = config.get_pipeline_config()
    ## If experiment data already exists, pass to next experiment
    if os.path.exists(os.path.join(pipeline_config.experiment_data_save_folder, "history.pickle")):
        return

    # Create System Modules
    system_rollout_config = config.get_system_rollout_config()
    system_rollout = create_system_rollout_module(system_rollout_config)
    rollout_statistics_encoder_config = config.get_rollout_statistics_encoder_config()
    rollout_statistics_encoder = create_rollout_statistics_encoder_module(rollout_statistics_encoder_config)

    # Create Intervention Modules
    intervention_config = config.get_random_intervention_generator_config()
    random_intervention_generator, intervention_fn = create_intervention_module(intervention_config)

    # Create Perturbation Modules
    perturbation_config = config.get_perturbation_generator_config()
    perturbation_generator, perturbation_fn = create_perturbation_module(perturbation_config)

    # Create IMGEP modules
    ## Goal Embedding Encoder, Generator and Achievement Loss
    goal_embedding_encoder_config = config.get_goal_embedding_encoder_config()
    goal_embedding_encoder = create_goal_embedding_encoder_module(goal_embedding_encoder_config)
    goal_generator_config = config.get_goal_generator_config()
    goal_generator = create_goal_generator_module(goal_generator_config)
    goal_achievement_loss_config = config.get_goal_achievement_loss_config()
    goal_achievement_loss = create_goal_achievement_loss_module(goal_achievement_loss_config)

    ## Goal-Conditioned Intervention Selector and Optimizer
    gc_intervention_selector_config = config.get_gc_intervention_selector_config()
    gc_intervention_selector = create_gc_intervention_selector_module(gc_intervention_selector_config)
    gc_intervention_optimizer_config = config.get_gc_intervention_optimizer_config()
    gc_intervention_optimizer = create_gc_intervention_optimizer_module(gc_intervention_optimizer_config)

    # Run
    run_imgep_experiment(pipeline_config.jax_platform_name, pipeline_config.seed,
                         pipeline_config.n_random_batches, pipeline_config.n_imgep_batches, pipeline_config.batch_size,
                         pipeline_config.experiment_data_save_folder,
                         random_intervention_generator, intervention_fn,
                         perturbation_generator, perturbation_fn,
                         system_rollout, rollout_statistics_encoder,
                         goal_generator, gc_intervention_selector, gc_intervention_optimizer,
                         goal_embedding_encoder, goal_achievement_loss,
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
        config = experiment_config.ExperimentConfig(model_idx, model_filepath, observed_node_ids, ymin, ymax, T, n_nodes)
        run_experiment(config)
