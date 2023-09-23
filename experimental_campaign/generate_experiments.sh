#!/bin/bash 

echo "Activate grn conda environment ..."
source ~/miniconda3/bin/activate grn

echo "Generate experiments ..."
python -c "import exputils
exputils.manage.generate_experiment_files('experiment_configurations.ods', directory='./experiments/', verbose=True)"

echo "Finished."

$SHELL
