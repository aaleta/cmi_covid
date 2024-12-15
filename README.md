# Informational approach to uncover the age group interactions in epidemic spreading from macro analysis

Code accompaning the paper *Informational approach to uncover the age group interactions in epidemic spreading from macro analysis* by Martinelli et al.

# Installation

This code was tested with Python 3.10. To install the dependencies, create a virtual environment and run `python -m pip install -r requirements.txt`. 

Then, clone the repository [IDTxl](https://github.com/pwollstadt/IDTxl) and install it with the following steps:

- Run from within the folder and with the virtual environment activated `python -m pip install .`

# Usage

The script `run_program.sh` executes all the analyses with the default values used in the paper. It is also possible to change each of those values as follows:

```commandline
usage: main.py [-h] [-s | --simulation | --no-simulation] [-se | --seir | --no-seir] [-n | --null | --no-null] [-w WAVE]
               [-gt | --generation-time | --no-generation-time] [-mt MAX_STEPS] [-b BOOTSTRAP_SAMPLES] [-l LOWER_CONFIDENCE] [-u UPPER_CONFIDENCE]

options:
  -h, --help            show this help message and exit
  -s, --simulation, --no-simulation
                        run a SIR model with input data
  -se, --seir, --no-seir
                        run a SEIR model with input data
  -n, --null, --no-null
                        override the input data with a null matrix in the SIR model
  -w WAVE, --wave WAVE  wave to study, should be between 1 and 6
  -gt, --generation-time, --no-generation-time
                        flag to indicate if past data should be aggregated by GT
  -mt MAX_STEPS, --max-steps MAX_STEPS
                        number of steps used in the aproximation of the generation time
  -b BOOTSTRAP_SAMPLES, --bootstrap-samples BOOTSTRAP_SAMPLES
                        number of bootstrap samples for the confidence interval estimation
  -l LOWER_CONFIDENCE, --lower-confidence LOWER_CONFIDENCE
                        lower confidence interval
  -u UPPER_CONFIDENCE, --upper-confidence UPPER_CONFIDENCE
                        upper confidence interval
```

Once finished, the script `plots_paper.py` will create the plots show in the paper.

# Notebooks

- matrix_aggregation: takes the original age contact matrix estimated by [Mistry et al.](https://doi.org/10.1038/s41467-020-20544-y), with 85 age groups, and aggregates it to the 10 age groups used by the Spanish Ministry of Health.

- data_preprocessing: segments the raw incidence as provided by the Spanish Ministry of Health into waves.


# How to cite

If you use this code, please cite the following paper:


Martinelli, T., Aleta, A., Rodrigues, F. A., & Moreno, Y. (2024). *Informational approach to uncover the age group interactions in epidemic spreading from macro analysis.* Physical Review E.
