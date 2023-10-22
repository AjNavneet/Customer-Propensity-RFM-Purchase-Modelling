# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings 
warnings.filterwarnings('ignore')

# Import custom functions and modules
from ml_pipeline.utils import read_config, read_data_csv, read_data_excel
from ml_pipeline.feature_eng import extract_features
from ml_pipeline.preprocessing import convert_to_datetime, clean_action_column, create_date_features, get_target_customers
from ml_pipeline.rfm import RFM_Features, RFMRanking, get_rfm_ranks
from ml_pipeline.preprocessing import filter_dataset_by_max_purchase_date, handling_missing_values
from ml_pipeline.model import ohe_categorical_feat, scale_features, split_into_train_test, train_baseline_logis_model, make_predictions, extract_important_features

# Reading the config file
config = read_config("D:/Testing/updated_code/modular_code/input/config.yaml")

# Reading our data
data = read_data_excel(config['data_path'])

# Correcting the datatype of the datetime column
data['DateTime'] = convert_to_datetime('DateTime', data)

# Generating RFM features for our dataset
RFM = RFM_Features(df=data[data['Action']=='purchase'], customerID="User_id", invoiceDate="DateTime", transID="Session_id", sales="Total Price")

# Getting RFM ranks, scores, and loyalty levels
RFM = get_rfm_ranks(RFM)

# Merging our main dataset with the RFM ranks data
data_with_RFM = pd.merge(data, RFM, on='User_id', how='left')

# Saving this combined dataset in the output folder
data_with_RFM.to_csv(config["final_data_path"], index=False)

# Reading the saved data again
data = read_data_csv(config["final_data_path"], parse_dates=['DateTime'])

# Cleaning the 'Action' column of the data
data = clean_action_column(data)

# Creating date-level features
data = create_date_features(data, 'DateTime')

# Making copies of the dataset
df = data.copy()

# Feature engineering
df_base = extract_features(df)

# Adding the target variable
# Filtering the dataset by the max purchase date for each user (All users who have done add_to_cart event)
df_base = filter_dataset_by_max_purchase_date(df, df_base)

# Handling null values
df_base = handling_missing_values(df_base, 'avg_time_between_purchase')

# Making a copy of the dataset
df_model = df_base.copy()

###### Model Building #####

# One-hot-Encoding
cat_df = ohe_categorical_feat(df_model)

# Scaling the numerical columns
num_df = scale_features(df_model)

# Merging categorical, numerical, and target variables
final_df = pd.concat([num_df, cat_df, df_model[['Target']], axis=1)

# Train-test split
x_train, x_test, y_train, y_test = split_into_train_test(final_df, 'Target')

# Training logistic regression baseline model
logreg = train_baseline_logis_model(x_train, y_train)

# Making predictions on the testing set
make_predictions(logreg, x_test, y_test)

# Display important features
extract_important_features(x_train, logreg)
