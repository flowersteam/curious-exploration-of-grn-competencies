# AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks


- Paper: [docs](https://developmentalsystems.org/curious-exploration-of-grn-competencies/index.html), [notebook](notebooks/paper.ipynb)
- Tutorial 1: [docs](https://developmentalsystems.org/curious-exploration-of-grn-competencies/tuto1.html), [notebook](notebooks/tuto1.ipynb)
- Tutorial 2: [docs](https://developmentalsystems.org/curious-exploration-of-grn-competencies/tuto2.html), [notebook](notebooks/tuto2.ipynb)

### Running the tutorials
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
The paper and tutorials are written in the form of jupyter notebooks and are placed in the `notebooks` folder. The complete structure of the repository is as follows 
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
