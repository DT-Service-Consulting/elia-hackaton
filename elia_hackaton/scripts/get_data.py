from elia_hackaton.core.utils import setup_gpu
from elia_hackaton.config import DATA_DIR
from elia_hackaton.core.extract_data import get_data_from_api, get_data_locally
from elia_hackaton.config import api_url, headers

import pandas as pd

device = setup_gpu()

# Load transformer parameters
tfo_parameters_df = pd.read_csv( DATA_DIR / 'tfo_parameters.csv', index_col=0)
#equipment = pd.read_csv(DATA_DIR / 'Equipment.csv', index_col=0)
print(tfo_parameters_df)

data_requested = 'equipment/GetAllTransformers/'

tfo_data = get_data_from_api(api_url, data_requested, headers)

if tfo_data != None:
    df_tfo = pd.DataFrame(tfo_data)
else:
    print('Error')

print("Success")

df_tfo_extra = pd.json_normalize(df_tfo['heatRunTest'])

df_tfo = pd.concat([df_tfo.drop(columns=['heatRunTest']), df_tfo_extra], axis=1)
df_tfo

df_tfo.to_csv('tfo_parameters.csv')


data_requested = 'equipment/GetAllTransformers/'

tfo_data = get_data_from_api(api_url, data_requested, headers)
df_tfo = pd.DataFrame(tfo_data)

print(df_tfo)
get_data_locally(df_tfo)
