# Author - Sahil Devgan| Term III  Project| May 1 2025


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Load the dataset
file_path = "Downloads/Marketing_Data.csv"  # Update path if needed
df = pd.read_csv(file_path)
# Quick check of data structure
df.info()
df.head(5)

# 2. Data Preparation and Feature Engineering
# Compute line-level purchase amount
df['LineAmount'] = df['Quantity'] * df['Price per Unit']
# Aggregate by Transaction ID to get transaction-level data
trans_df = df.groupby('Transaction ID').agg({
    'Customer ID': 'first',
    'Date': 'first',
    'Gender': 'first',
    'Age': 'first',
    'Quantity': 'sum',        # total quantity in transaction
    'LineAmount': 'sum',      # total amount in transaction (DEPVAR)
    'Product Category': lambda cats: list(cats)  # list of categories bought
}).reset_index()
# Rename aggregated columns for clarity
trans_df.rename(columns={'Quantity': 'TotalQuantity', 'LineAmount': 'TotalAmount'}, inplace=True)

# Create indicator (dummy) variables for presence of each product category in the transaction
trans_df['Has_Beauty'] = trans_df['Product Category'].apply(lambda cat_list: 1 if 'Beauty' in cat_list else 0)
trans_df['Has_Clothing'] = trans_df['Product Category'].apply(lambda cat_list: 1 if 'Clothing' in cat_list else 0)
trans_df['Has_Electronics'] = trans_df['Product Category'].apply(lambda cat_list: 1 if 'Electronics' in cat_list else 0)

# Encode Gender as binary (e.g., Female=1, Male=0)
trans_df['GenderBinary'] = trans_df['Gender'].map({'Male': 0, 'Female': 1})

# 3. Exploratory Data Analysis (EDA)
# Summary statistics for key numerical variables
num_vars = ['Age', 'Quantity', 'Price per Unit']
print("Summary of numerical variables (line-item level):")
print(df[num_vars].describe())

print("\nSummary of target variable (TotalAmount) per transaction:")
print(trans_df['TotalAmount'].describe())

# Correlation matrix (for transaction-level data)
corr_vars = ['TotalAmount', 'Age', 'GenderBinary', 'TotalQuantity']
corr_matrix = trans_df[corr_vars].corr()
print("\nCorrelation matrix (transaction-level):")
print(corr_matrix)

# Analyze average spending by categories, gender, etc.
print("\nAverage TotalAmount by Gender:")
print(trans_df.groupby('Gender')['TotalAmount'].mean())
print("\nAverage TotalAmount by number of product categories in transaction:")
cat_count = trans_df['Product Category'].apply(lambda x: len(set(x)))  # unique categories count
trans_df['NumCategories'] = cat_count
print(trans_df.groupby('NumCategories')['TotalAmount'].mean())
print("\nAverage TotalAmount by age group:")
trans_df['AgeGroup'] = pd.cut(trans_df['Age'], bins=[18,25,35,45,55,65], right=False)
print(trans_df.groupby('AgeGroup')['TotalAmount'].mean())

# 4. Modeling: Multiple Linear Regression
# Define predictor matrix X and target vector y for the regression (using selected features)
X = trans_df[['GenderBinary', 'Age', 'TotalQuantity', 'Has_Beauty', 'Has_Clothing', 'Has_Electronics']]
y = trans_df['TotalAmount']
# Add constant term for intercept
X = sm.add_constant(X)
# Fit initial model with all candidate predictors
model = sm.OLS(y, X).fit()
print("\nInitial model summary:")
print(model.summary())

# Refine model by removing insignificant predictors (Gender and category indicators in this case)
X_final = trans_df[['Age', 'TotalQuantity']]  # selected best predictors
X_final = sm.add_constant(X_final)
final_model = sm.OLS(y, X_final).fit()
print("\nFinal model summary:")
print(final_model.summary())

# 5. Model Evaluation
# Calculate R² and RMSE for the final model
r_squared = final_model.rsquared
rmse = np.sqrt(np.mean((final_model.fittedvalues - y) ** 2))
print(f"\nFinal Model R²: {r_squared:.3f}")
print(f"Final Model RMSE: {rmse:.2f}")

# Plot Actual vs Predicted values for Total Amount
plt.figure(figsize=(6,6))
plt.scatter(y, final_model.fittedvalues, alpha=0.6)
plt.plot([0, trans_df['TotalAmount'].max()], [0, trans_df['TotalAmount'].max()], 
         color='red', linestyle='--', label='Ideal: Predicted = Actual')
plt.xlabel("Actual Total Amount")
plt.ylabel("Predicted Total Amount")
plt.title("Actual vs Predicted Total Amount per Transaction")
plt.legend()
plt.tight_layout()
plt.savefig('Actual_vs_Predicted.png')  # Save the plot as an image file

# (The saved plot image will be used in the report.)
