# Importing required packages
from datetime import timedelta
import pandas as pd

# Defining a function that can be used to generate RFM features
def RFM_Features(df, customerID, invoiceDate, transID, sales):
    ''' Create the Recency, Frequency, and Monetary features from the data '''
    # Final date in the data + 1 to create the latest date
    latest_date = df[invoiceDate].max() + timedelta(1)
    
    # RFM feature creation
    RFMScores = df.groupby(customerID).agg({invoiceDate: lambda x: (latest_date - x.max()).days, 
                                          transID: lambda x: len(x), 
                                          sales: lambda x: sum(x)})
    
    # Converting invoiceDate to int since this contains the number of days
    RFMScores[invoiceDate] = RFMScores[invoiceDate].astype(int)
    
    # Renaming column names to Recency, Frequency, and Monetary
    RFMScores.rename(columns={invoiceDate: 'Recency', 
                         transID: 'Frequency', 
                         sales: 'Monetary'}, inplace=True)
    
    return RFMScores.reset_index()

# Defining a function for creating RFM ranks
def RFMRanking(x, variable, quantile_dict):
    ''' Ranking the Recency, Frequency, and Monetary features based on quantile values '''
    
    # Checking if the feature to rank is Recency
    if variable == 'Recency':
        if x <= quantile_dict[variable][0.25]:
            return 4
        elif (x > quantile_dict[variable][0.25]) & (x <= quantile_dict[variable][0.5]):
            return 3
        elif (x > quantile_dict[variable][0.5]) & (x <= quantile_dict[variable][0.75]):
            return 2
        else:
            return 1
    
    # Checking if the feature to rank is Frequency and Monetary
    if variable in ('Frequency','Monetary'):
        if x <= quantile_dict[variable][0.25]:
            return 1
        elif (x > quantile_dict[variable][0.25]) & (x <= quantile_dict[variable][0.5]):
            return 2
        elif (x > quantile_dict[variable][0.5]) & (x <= quantile_dict[variable][0.75]):
            return 3
        else:
            return 4

# Defining a function for creating quantiles, individual and grouped RFM ranks and loyalty levels
def get_rfm_ranks(rfm_feature_dataset):

    # Creating quantiles
    Quantiles = rfm_feature_dataset[['Recency', 'Frequency', 'Monetary']].quantile([0.25, 0.50, 0.75])
    Quantiles = Quantiles.to_dict()
    
    # Individual RFM ranks 
    rfm_feature_dataset['R'] = rfm_feature_dataset['Recency'].apply(lambda x: RFMRanking(x, variable='Recency', quantile_dict=Quantiles))
    rfm_feature_dataset['F'] = rfm_feature_dataset['Frequency'].apply(lambda x: RFMRanking(x, variable='Frequency', quantile_dict=Quantiles))
    rfm_feature_dataset['M'] = rfm_feature_dataset['Monetary'].apply(lambda x: RFMRanking(x, variable='Monetary', quantile_dict=Quantiles)
   
    # Combined RFM Scores
    rfm_feature_dataset['Group'] = rfm_feature_dataset['R'].apply(str) + rfm_feature_dataset['F'].apply(str) + rfm_feature_dataset['M'].apply(str)
    rfm_feature_dataset["Score"] = rfm_feature_dataset[['R', 'F', 'M']].sum(axis=1)

    # Loyalty levels
    loyalty = ['Bronze', 'Silver', 'Gold', 'Platinum']
    rfm_feature_dataset['Loyalty_Level'] = pd.qcut(rfm_feature_dataset['Score'], q=4, labels=loyalty)

    return rfm_feature_dataset
