# Importing required packages
import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ml_pipeline.utils import read_config

# Reading the config file
config = read_config("D:/Testing/updated_code/modular_code/input/config.yaml")

# Function for One-hot-Encoding the categorical data
def ohe_categorical_feat(data):
    cat_cols = ['top_paths'] # Only top_paths feature
    cat_df = data[cat_cols]
    cat_df = pd.get_dummies(cat_df)
    return cat_df

# Function for Scaling the numeric data
def scale_features(data):
    scaler = MinMaxScaler()
    num_cols = data.drop(['Target','User_id'],axis=1).select_dtypes(['int','float']).columns.tolist()
    num_df = scaler.fit_transform(data[num_cols])
    num_df = pd.DataFrame(num_df, columns = num_cols)
    return num_df

# Function for Train test split
def split_into_train_test(data, target_variable):
    X = data.drop([target_variable],axis=1)
    Y = data[[target_variable]]
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=config['test_size'])
    return x_train, x_test, y_train, y_test

# Function for Training a baseline logistic regression model
def train_baseline_logis_model(x_train, y_train):
    logreg = LogisticRegression(class_weight='balanced',random_state=config['random_state'])
    logreg.fit(x_train,y_train)
    return logreg

# Function to make predictions on the testing set
def make_predictions(logist_reg_model, x_test, y_test):
    preds = logist_reg_model.predict_proba(x_test)[:,1]
    print(preds)
    print("Test ROC-AUC:" + str(metrics.roc_auc_score(y_test,preds))

    preds_label = logist_reg_model.predict(x_test)
    print(preds_label)
    print("Test Accuracy:" + str(metrics.accuracy_score(y_test,preds_label)))

# Function to extract important features
def extract_important_features(x_train,logistic_reg_model):
    print(pd.DataFrame(zip(x_train.columns,logistic_reg_model.coef_[0]), columns=['Feats','Imp']).sort_values(by='Imp',ascending=False))
