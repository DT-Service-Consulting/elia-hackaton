from elia_hackaton.core.utils import setup_gpu
from elia_hackaton.config import DATA_DIR

import pandas as pd

device = setup_gpu()

# Load transformer parameters
tfo_parameters_df = pd.read_csv( DATA_DIR / 'tfo_parameters.csv')
#equipment = pd.read_csv(DATA_DIR / 'Equipment.csv', index_col=0)
print(tfo_parameters_df)


exit()
