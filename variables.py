import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from variables import *
from sklearn.preprocessing import MinMaxScaler
import os

# GENERAL DIRECTORY DATA
grid_data_folder_name="grids_data"
full_current_grids_data_dir=f'./{grid_data_folder_name}'
models_folder='./models_trained'
grid_analysis_folder='grid_analysis'
full_report_folder='full_report'
eval_folder='evaluation_per_model'

underconsume_min=0.3
underconsume_max=0.5
overconsume_min=1.2
overconsume_max=2

underconsume_population=30
overconsume_population=20
broken=15

# GENERAL DATA PROCESSING PARAMETERS
synth_data_columns_headers=['node_id','energy','time']
filename_synth_data='synth_energy_data.csv'
filename_report_effect_broken_issues='report_effect_broken_issues.csv'

# ISSUES DATA
noise_percentage=0.1
issue_duration_mean=120 # in minutes

# GET TIMESTEP AND WINDOW TIME RANGE
max_days=15

minutes_per_point=10 # It cannot be 0

points_per_min=1/minutes_per_point
max_time_steps=int(max_days*24*60*points_per_min) # If every step is 10 minutes, it is equivalent to a whole week of data

all_steps=np.arange(0,max_time_steps) # Every stept time in integer indexes unit.

# INIT ENERGY DATA
init_energy_data=pd.DataFrame(columns=synth_data_columns_headers)

# GRID TEMPLATE
max_sub_grids=1
min_sub_grids=1

max_consumer_nodes=40
min_consumer_nodes=20

max_producer_nodes=3
min_producer_nodes=1

min_consumer_energy=60
max_consumer_energy=100

max_degree=2
min_degree=1

num_grids=8

# TRAINING AND DETECTING
train_test_ratio=10
model_name='rf_issue_detect_model.sav'
synth_columns_grid_data=['node_id','energy','time','issue_type','starting_time','rootcause','node_type']