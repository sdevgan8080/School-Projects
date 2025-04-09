# @Author - Sahil Devgan| Assignment 09 | April 09-2025. 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load the dataset
file_path = "Downloads/bill_all.csv"  
#df = pd.read_csv(file_path)
df = pd.read_csv(file_path, encoding='ISO-8859-1')


# 2. Data Cleaning
# Convert Date to datetime object
df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%y")
# Drop or fill any irrelevant or missing data fields (if any)
df = df.drop(columns=['Promo_Stat'], errors='ignore')  # drop Promo_Stat as it's mostly missing

# 3. Basic Data Inspection
print(df.shape)         # rows, columns
print(df.columns.tolist())
print(df[['Bill_No','Date','Category','Sub_Category','Item_Descp','Qty','SP','Net_sales']].head(5))

# 4. Exploratory Data Analysis

# Number of transactions and items
num_transactions = df['Bill_No'].nunique()
total_items_sold = len(df)
avg_items_per_trans = total_items_sold / num_transactions
print("Total transactions:", num_transactions)
print("Total items sold:", total_items_sold)
print("Average items per transaction:", avg_items_per_trans)

# Distribution of basket sizes (items per transaction)
basket_sizes = df.groupby('Bill_No').size()  # count of rows per Bill_No
print(basket_sizes.describe())  # summary stats (mean, min, max, quartiles)
# Histogram of basket sizes
plt.figure(figsize=(6,4))
plt.hist(basket_sizes, bins=range(1, 21), edgecolor='black', align='left')
plt.xlabel('Items per Transaction')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Basket Sizes')
plt.xticks(range(1,21))
plt.tight_layout()
plt.show()

# Top categories by frequency and sales
category_counts = df['Category'].value_counts()
category_sales = df.groupby('Category')['Net_sales'].sum().sort_values(ascending=False)
print("Top 5 Categories by Item Count:\n", category_counts.head(5))
print("Top 5 Categories by Sales (Revenue):\n", category_sales.head(5))

# Bar chart of top 10 categories by frequency
top10 = category_counts.head(10)
plt.figure(figsize=(6,4))
top10.sort_values().plot(kind='barh', color='steelblue')
plt.xlabel('Number of Items Sold')
plt.title('Top 10 Categories by Frequency of Items Sold')
plt.tight_layout()
plt.show()

# Transactions by weekday (optional analysis)
df['WeekdayName'] = df['Date'].dt.day_name()
trans_per_day = df.groupby('WeekdayName')['Bill_No'].nunique()
print("Transactions per weekday:\n", trans_per_day)

# 5. Conjoint Analysis Data Preparation

# Focus on single-item transactions for choice modeling
# Find all Bill_No that have only 1 item
transaction_counts = df['Bill_No'].value_counts()
single_item_bills = transaction_counts[transaction_counts == 1].index
single_trans_df = df[df['Bill_No'].isin(single_item_bills)].copy()

# Create choice sets for each single-item transaction
# For each single transaction, get the chosen item attributes and sample 4 alternatives
import random
random.seed(42)
# Prepare a list of all unique item codes to sample alternatives from
all_item_codes = df['Item_code'].unique().tolist()

choice_data = []  # to collect choice scenarios
scenario_id = 0
for bill, record in single_trans_df.iterrows():
    scenario_id += 1
    chosen_code = record['Item_code']
    chosen_category = record['Category']
    chosen_price = record['SP']
    # Assign price level based on bins
    if chosen_price <= 50:
        chosen_price_level = 'Low'
    elif chosen_price <= 150:
        chosen_price_level = 'Medium'
    else:
        chosen_price_level = 'High'
    # Add the chosen product (outcome = 1)
    choice_data.append({
        'Scenario': scenario_id,
        'Chosen': 1,
        'Category': chosen_category,
        'Price_Level': chosen_price_level
    })
    # Sample 4 alternative item codes that are not the chosen item
    alt_codes = random.sample([code for code in all_item_codes if code != chosen_code], 4)
    for alt_code in alt_codes:
        alt_record = df[df['Item_code'] == alt_code].iloc[0]  # take first occurrence of that item
        alt_category = alt_record['Category']
        alt_price = alt_record['SP']
        if alt_price <= 50:
            alt_price_level = 'Low'
        elif alt_price <= 150:
            alt_price_level = 'Medium'
        else:
            alt_price_level = 'High'
        choice_data.append({
            'Scenario': scenario_id,
            'Chosen': 0,
            'Category': alt_category,
            'Price_Level': alt_price_level
        })

choice_df = pd.DataFrame(choice_data)
print("Choice data sample:\n", choice_df.head())

# Ensure Category and Price_Level are treated as categorical and set baseline levels
# Choose a baseline category (one with low frequency, e.g., "HOUSEHOLD PLACTIC")
baseline_category = "HOUSEHOLD PLACTIC"
if baseline_category not in choice_df['Category'].unique():
    baseline_category = choice_df['Category'].value_counts().idxmin()  # choose least frequent as baseline
# Set categorical dtype with specified baseline (first category)
cat_categories = [baseline_category] + [cat for cat in choice_df['Category'].unique() if cat != baseline_category]
choice_df['Category'] = pd.Categorical(choice_df['Category'], categories=cat_categories)
# Set Price_Level categorical with Low as baseline
choice_df['Price_Level'] = pd.Categorical(choice_df['Price_Level'], categories=['Low','Medium','High'], ordered=True)

# 6. Conjoint (Logistic Regression) Model
model = smf.logit(formula='Chosen ~ C(Category) + C(Price_Level)', data=choice_df).fit()
print(model.summary())  # Regression summary with coefficients

# Extract part-worth utilities from model coefficients
params = model.params
# Baseline levels have no explicit coefficient (captured in intercept), we will set their utility to 0
baseline_cat = baseline_category
baseline_price = 'Low'
# Calculate category utilities
cat_utils = {baseline_cat: 0.0}
for lvl in cat_categories:
    if lvl == baseline_cat:
        continue
    coeff_name = f"C(Category)[T.{lvl}]"
    if coeff_name in params:
        cat_utils[lvl] = params[coeff_name]
# Calculate price level utilities
price_utils = {baseline_price: 0.0}
for lvl in ['Medium','High']:
    coeff_name = f"C(Price_Level)[T.{lvl}]"
    if coeff_name in params:
        price_utils[lvl] = params[coeff_name]

print("Part-worth utilities for Category (baseline = {}):".format(baseline_cat))
for cat, util in cat_utils.items():
    print(f"  {cat}: {util:.3f}")
print("Part-worth utilities for Price Level (baseline = Low):")
for lvl, util in price_utils.items():
    print(f"  {lvl}: {util:.3f}")

# Calculate relative importance of Category vs Price
cat_range = max(cat_utils.values()) - min(cat_utils.values())
price_range = max(price_utils.values()) - min(price_utils.values())
importance_cat = (cat_range / (cat_range + price_range)) * 100
importance_price = (price_range / (cat_range + price_range)) * 100
print(f"Estimated attribute importance -> Category: {importance_cat:.1f}% , Price: {importance_price:.1f}%")

# 7. Visualization of part-worth results (optional)
# Price level part-worth bar chart
price_util_series = pd.Series(price_utils)
plt.figure(figsize=(4,3))
price_util_series.reindex(['Low','Medium','High']).plot(kind='bar', color=['#4caf50','#ffa726','#ff7043'])
plt.xlabel('Price Level')
plt.ylabel('Utility')
plt.title('Part-Worth Utilities for Price Levels')
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

# Category part-worth bar chart (sorted)
cat_util_series = pd.Series(cat_utils)
plt.figure(figsize=(8,6))
cat_util_series.sort_values(ascending=False).plot(kind='barh', color='skyblue')
plt.xlabel('Utility')
plt.title('Category Part-Worth Utilities (Baseline = {})'.format(baseline_cat))
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()
