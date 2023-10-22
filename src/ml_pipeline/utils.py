# Importing required packages
import pandas as pd
import numpy as np
from yaml import CLoader as Loader, load
import warnings
warnings.filterwarnings('ignore')

# Function to read a csv file
def read_data_csv(file_path, **kwargs):
    # Read the CSV file using pandas and any additional keyword arguments
    raw_data_csv = pd.read_csv(file_path, **kwargs)
    return raw_data_csv

# Function to read an Excel file
def read_data_excel(file_path, **kwargs):
    # Read the Excel file using pandas and any additional keyword arguments
    raw_data_excel = pd.read_excel(file_path, **kwargs)
    return raw_data_excel

# Function for reading a config file
def read_config(path):
    # Read the YAML configuration file using PyYAML
    with open(path) as stream:
        config = load(stream, Loader=Loader)
    return config
