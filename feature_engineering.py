# %%
# Import useful libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# %% Directory setup
# Set relative path to the data file
base_dir = os.path.dirname(os.path.abspath(__file__))
toyota_path = os.path.join(base_dir, "data", "toyota_cleaned.csv")
# %%
toyota = pd.read_csv(toyota_path) # Insert the dataset as a pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns of the dataset
print("The dimensions of the cleaned dataset are:", toyota.shape) # Dimensions of the dataset
print(toyota.head()) # First 5 rows of the dataset
# %% Air condition
# Simplify and avoid multicollinearity concerns
toyota['AC'] = toyota['Aircondition'] + toyota['Automatic_Aircondition'] # 0 for no AC, 1 for manual AC, 2 for automatic AC
toyota = toyota.drop(columns=['Aircondition', 'Automatic_Aircondition'])
# %% Airbags
# Simplify and avoid multicollinearity concerns
toyota["Airbags_Number"] = toyota['Airbag_1'] + toyota['Airbag_2'] 
toyota = toyota.drop(columns=['Airbag_1', 'Airbag_2'])
# %% Gears
print("The number of cars in each Gears category is:\n", toyota['Gears'].value_counts()) # We see most cars are 5th gear
print('The average price per category is:\n', toyota.groupby('Gears')['Price'].mean()) # The real difference in price is between 5 and 6 gears. Also 3 and 4 gears have few observations, so I merge them with 5 gears.

toyota['Sixth_Gear'] = toyota['Gears'].apply(lambda x: 1 if x == 6 else 0) # Create indicator variable for cars with 6 gears
toyota = toyota.drop(columns=['Gears']) 
# %% Guarantee Period
print("The number of cars in each Guarantee Period category is:\n", toyota['Guarantee_Period'].value_counts()) # we see that most cars have 3 months of guarantee. Then above 24 months, we have very few cars.
print('The average price per category  is:\n', toyota.groupby('Guarantee_Period')['Price'].mean()) # there is a jump in price for cars with above 12 months of guarantee.

# Create long Guarantee indicator
toyota['Long_Guarantee'] = toyota['Guarantee_Period'].apply(lambda x: 1 if x > 12 else 0) # create indicator variable for cars with long guarantee
toyota = toyota.drop(columns=['Guarantee_Period'])
# %% Quarterly Tax
print("The number of cars in each Quarterly_Tax category is:\n", toyota['Quarterly_Tax'].value_counts()) 
print('The average price per category e is:\n', toyota.groupby('Quarterly_Tax')['Price'].mean())

# Check some stats aggregating to help decide on a threshold
tax_stats = toyota.groupby('Fuel_Type')['Quarterly_Tax'].agg(['mean', 'min', 'max', 'count'])
print(tax_stats)

# Create indicator variable for High_Tax vs Low_Tax cars. This will include also info if the car is Diesel/CNG or Petrol. 
# Then drop Fuel_Type column to avoid multicollinearity issue because it is closely related to quarterly taxes.
toyota['High_Tax'] = toyota['Quarterly_Tax'].apply(lambda x: 1 if x > 170 else 0)
toyota = toyota.drop(columns=['Quarterly_Tax'])
toyota = toyota.drop(columns=['Fuel_Type'])
# %% Doors
print("The number of cars  in each Doors category is:\n", toyota['Doors'].value_counts()) # Only 2 cars with 2 doors, outliers
print('The average price per category e is:\n', toyota.groupby('Doors')['Price'].mean())
print('we see that cars with 3 and 4 doors have similar average prices, while cars with 5 doors are more expensive on average.')

# Create an indicator variable for cars with 5 doors
toyota['Is_5_Doors'] = toyota['Doors'].apply(lambda x: 1 if x == 5 else 0) # Create indicator variable for cars with 5 doors
toyota = toyota.drop(columns=['Doors']) # Drop the original Doors column
# %% Parking Assistant
print("The number of cars with a parking assistant in each category is:\n", toyota['Parking_Assistant'].value_counts()) # only 4 cars with a parking assistant
toyota = toyota.drop(columns=['Parking_Assistant']) # I decide to drop this variable to reduce noise
# %% Color
most_frequent_color = toyota['Color'].value_counts().idxmax() # Find the most common color
toyota = pd.get_dummies(toyota, columns=['Color'], dtype = int) # Create dummies for all colors
toyota = toyota.drop(columns=[f'Color_{most_frequent_color}']) # Drop the most frequent color to avoid dummy trap
# %% Correlation Matrix with Pearson's Coefficient
correlation_mat = toyota.corr(method = 'pearson')
np.fill_diagonal(correlation_mat.values, np.nan) # remove correlation of price with itself

mask = np.triu(np.ones_like(correlation_mat, dtype=bool)) # generate a mask for the upper triangle

# Plot
plt.figure(figsize=(22, 12))
sns.heatmap(
    correlation_mat,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f",
    annot_kws={"size": 7}
)

plt.title("Heatmap of the Lower Triangle of Correlation Matrix with Pearson's Coefficient")

plt.savefig("graphs/Correlation Matrix_Linear Modeling.png", dpi=300, bbox_inches="tight")
plt.close() 
# %% Variance Inflation Factor (VIF) to check for multicollinearity

# As we saw on the correlation matrix, some features are highly correlated. 
# VIF helps us quantify how much the variance of a regression coefficient is inflated due to multicollinearity among the features.
# If VIF > 5, we have multicollinearity issues. So we need to consider dropping some features or combining them.
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = toyota.drop(columns=['Price'])  # Drop response variable
X = X.assign(const=1)           # Add intercept

vif = pd.DataFrame() # Create empty dataframe to store VIF values
vif["Feature"] = X.columns

vif_list = [] # Empty list to store VIF values in the loop

for i in range(X.shape[1]):
    
    value = variance_inflation_factor(X.values, i)    
    vif_list.append(value)

vif["VIF"] = vif_list # Assign the VIF values to the dataframe

print('The VIF values are:\n', vif.sort_values("VIF", ascending=False).round(2)) 

# Rule of Thumb: VIF > 5 indicates multicollinearity issues. 
# We do not care about the constant term.
# %% Save the cleaned and preprocessed dataset to a new CSV file for Linear Modeling
toyota.to_csv('data/toyota_ready_for_modeling.csv', index=False)
print(toyota.head().to_string())
print("The dimesions of the cleaned and preprocessed dataset for Linear Modeling are:", toyota.shape)

# %% End of Code