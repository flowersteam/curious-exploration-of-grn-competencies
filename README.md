# AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks

This repository hosts the source code to reproduce the results presented in the paper [AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks](https://osf.io/s6thq/):

- Paper: [docs](https://developmentalsystems.org/curious-exploration-of-grn-competencies/index.html), [notebook](notebooks/paper.ipynb)
- Tutorial 1: [docs](https://developmentalsystems.org/curious-exploration-of-grn-competencies/tuto1.html), [notebook](notebooks/tuto1.ipynb)
- Tutorial 2: [docs](https://developmentalsystems.org/curious-exploration-of-grn-competencies/tuto2.html), [notebook](notebooks/tuto2.ipynb)



## Running the notebooks

Clone the repository on your local machine 
```sh
git clone https://github.com/flowersteam/curious-exploration-of-grn-competencies.git
```
The code comes with an anacoda environment that has all the requirements pre-installed.
You can create the environment using 

```sh
conda env create -f env.yaml # creating the environment
```
Then you can activate the environment and visualise the paper and the tutorials using the jupyter lab
```sh
conda activate curious_assistant
jupyter lab
```
The paper and tutorials are written in the form of jupyter notebooks and are placed in the `notebooks` folder. The complete structure of the repository is as follows: 
```sh
├── README.md
├── LICENSE
├── env.yaml                        # anaconda environment
│
├── notebook_to_html_export.ipynb   # notebook generating the docs
├── docs                            # generated docs html
│
├── experimental_campaign           # data and code for the experiments
│
└── notebooks                       # jupyter notebooks
   ├── figures
   ├── paper.ipynb                  # paper notebook
   ├── tuto1.ipynb                  # tutorial 1 notebook
   └── tuto2.ipynb                  # tutorial 2 notebook
```   
 

### Reproduce the paper results

Before trying to regenerate the paper plots please unzip the `experiment_data_statistics.zip` and `evaluation_data_examples.zip` files in the `experimental_campaign/analysis/` folder.

Then you can just run the `paper.ipynb` notebook. 

### Running the tutorials
As some of the tutorial cells might take considerable amount of time to execute the tutorial notebooks come with an option to load already executed data, which is much faster. This functionality can be used as an initiation to the tutorials, when running them for the first time.

```python
nb_mode = "load" #@param ["run", "load"]
```
If you want to regenerate all the data and run the tutorial examples set the `nb_mode` to
```python
nb_mode = "run" #@param ["run", "load"]
```

If you want to save the newly generated figures (so that they are updated as well in `paper.ipynb`) set
```python
nb_save_figs = True #@param {type:"boolean"}
```
### Running the experimental campaign

For running the experimental campaign, activate the conda environment and navigate into the corresponding folder:

```sh
conda activate curious_assistant
cd experimental_campaign/
```

The `experimental_campaign` folder structure is as follows
```sh
├── resources               # database creation
   ├── *.py                 # python scripts to generate the database
   ├── bio_models           # folder containing the biological network models
   └── *.csv and *.npy      # statistics about the databases
├── experiments             # running experiments
   ├── experiment_000001    # python scripts to run the random exploration baseline
   ├── experiment_000003    # python scripts to run the curiosity-driven exploration and robustness tests
└── analysis                # collect the data for analysis
   ├── *.py                 # python scripts to gather data necessary for notebooks
   └── *.pickle             # analysis data
```   

Please note that reproducing the whole experimental campaign over the 432 systems needs a long time to compute and can take a lot of space (>500GB). The final analysis data (~350MB) needed to reproduce the paper main figures is already provided as pickle files in the `experimental_campaign/analysis/` folder (after unzipping). For re-generating that data from scratch, you can follow the below steps (1-2-3) but we recommend running the campaign on supercomputers if possible, and then transferring back only the analysis data on local computer.

#### 1) Database creation
The database of biological networks used in the paper is contained in the `experimental_campaign/resources/` folder, go in it as follows:
 ```sh
cd resources/
 ```
 
You can delete the `experimental_campaign/resources/bio_models` folder as well as all the `experimental_campaign/resources/*.csv` files and regenerate them as follows:


```python
python run_bio_models_preselection.py
python run_bio_models_nodes_selection.py
```

The database of random networks as well as the analysis of their versatility (Figure 7 of the main paper) can also be reproduced by deleting the `random_networks_versatility.npy` file and regenerating it as follows:

```python
python generate_random_models.py
```

#### 2) Running exploration algorithms and robustness tests
To run the main experiments, go in the corresponding folder as follows:
 ```sh
cd ../experiments/
 ```

For running the curiosity-driven exploration algorithm on the database of biological networks do:

```python
cd experiment_000003/
python run_experiment.py
```

When finished, you can run the robustness tests as follows:
```python
python run_evaluation.py
```

You can also run the random search exploration baseline as follows:
```python
cd experiment_000001/
python run_experiment.py
```


#### 3) Collect the data for analysis
Finally, you can gather the "analysis data" from the data generated in step 2 as follows:
```python
cd ../../analysis
python gather_experiment_data_statistics.py #exploration algorithms data
python gather_evaluation_data_statistics.py #robustness tests data
python gather_evaluation_data_examples.py #robustness tests examples for Figure 5
```

These final steps output `pickle` files that are used to run the paper notebook. 

## Contact 
If you have any questions about the notebooks or the paper feel free to contact me at the email: mayalen.etcheverry@inria.fr
