from autodiscjax.experiment_pipelines import run_rs_experiment
from autodiscjax.utils.create_modules import *
import csv
import experiment_config
import jax.numpy as jnp
import os

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

    # Run
    run_rs_experiment(pipeline_config.jax_platform_name, pipeline_config.seed,
                         pipeline_config.n_random_batches, pipeline_config.batch_size,
                         pipeline_config.experiment_data_save_folder,
                         random_intervention_generator, intervention_fn,
                         perturbation_generator, perturbation_fn,
                         system_rollout, rollout_statistics_encoder,
                         out_sanity_check=True, save_modules=False, save_logs=True)


if __name__ == "__main__":

    database_stats_filepath = "../../resources/bio_models_database.csv"
    with open(database_stats_filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        model_idx = int(row["model_idx"])
        model_filepath = f"../../resources/bio_models/model_{model_idx:06d}.py"
        ymin = jnp.array(eval(row["ymin"]))
        ymax = jnp.array(eval(row["ymax"]))
        T = int(row["T"])
        n_nodes = int(row["n_nodes"])

        print(model_idx)
        config = experiment_config.ExperimentConfig(model_idx, model_filepath, ymin, ymax, T, n_nodes)
        run_experiment(config)
