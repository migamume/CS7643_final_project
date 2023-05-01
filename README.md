# CS7643_final_project
For our course' final project

# Running locally
This repository is now an installable python package. You should activate your virtualenv / conda environment (TODO: Conda) and then install it for local development like so:

`pip install -e ./`

This will install a package named `holdup` in your python environment.

### Logging Note
David reused some old logging code that expects your environment to have a `LOG_CFG_PATH` specified. This should point to the `logcfg.yaml` file. I could probably have made this a resource instead, but I haven't gotten around to it. I modified the code so it falls back to a base logger if you haven't set up this env var. But if you want to use the logging config, run this from terminal:

`export LOG_CFG_PATH='/path/to/CS7643_final_project/logcfg.yaml'

for me this is

`export LOG_CFG_PATH='/home/david/development/omscs/dl/CS7643_final_project/logcfg.yaml'

## From terminal
After you have installed the package for local development, run as you would any other python file.

### Downloading data
Rather than commit GBs of data, I setup a gitignore to ignore the data dirs. Each of us will need to download and parse the same datasets.

To do so, run the `data_prepper` like so:

`python src/holdup/parser/data_prepper.py`

## Development in Notebooks
If you use jupyter notebooks for development, you can now install the package and then import from it like any other.

Here's how to make your conda env available in Jupyter: https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

### More ReadMe's
Can be found in src/holdup -> the_model and visualizations