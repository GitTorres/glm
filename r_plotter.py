from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

r_utils = importr('utils')
r_tidy = importr('tidyverse')
r_arrow = importr('arrow')

# r_utils.install_packages('tidyverse')
# r_utils.install_packages('arrow')