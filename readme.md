# Customer Propensity to RFM Purchase Modelling 

## Business Objective
Our Client is an early-stage e-commerce company selling various products from daily essentials to high-end electronics and home appliances. They aim to increase purchases by sending discounts or coupons to users based on a predictive model that estimates purchase probability.

---

## Objective
This project focuses on building a Propensity to Purchase Model using Python, with a primary objective of improving user engagement and ROI. We employ Propensity Modeling and RFM (Recency, Frequency, Monetary) Analysis to predict users' likelihood of making a purchase and to identify high-value customer segments.

---

## Data Description
The dataset contains purchase history data for an e-commerce company over a period of time.

---

## Aim
1. Understand Propensity Modeling.
2. Understand RFM Analysis.
3. Build a model to predict the purchase probability of each user in an e-commerce company using the Propensity Model.

---

## Tech Stack
- Language: `Python`
- Libraries: `pandas`, `scikit-learn`, `numpy`, `seaborn`, `datetime`, `matplotlib`, `missingno`

---

## Approach
1. Import the required libraries and packages.
2. Read the CSV file.
3. Perform data preprocessing.
4. Conduct exploratory data analysis.
   - Univariate analysis
   - Multivariate analysis
1. Perform RFM Analysis.
2. Perform feature engineering.
3. Create modeling data.
4. Build the predictive model.
5. Make predictions.

---

## Code Structure
- `input`: Contains data and configuration files.
   - `config.yaml`: Configuration parameters.
   - `final_customer_data.xlsx`: Customer transaction data.
   - `final_customer_data_with_RFM_features.csv`: Merged dataset with RFM values.
   - `ecom_product_data.csv`: Transaction data for RFM modeling.
- `src`: Contains modularized code for different project steps.
   - `engine.py`: Main execution script.
   - `ml_pipeline`: Modular functions.
   - `requirements.txt`: List of required packages.
- `output`: Stores the trained model for future use.
- `lib`: Reference notebooks from the project.

---


