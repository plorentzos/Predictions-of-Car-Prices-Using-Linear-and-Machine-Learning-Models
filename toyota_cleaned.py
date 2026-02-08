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
toyota_path = os.path.join(base_dir, "data", "ToyotaCorolla.csv")
# %% Data Inspection and Cleaning
toyota = pd.read_csv(toyota_path) # Insert the dataset as a pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns of the dataset
print(toyota.shape) # Dimensions of the dataset
print(toyota.head().to_string()) # First 5 rows of the dataset
# %% Columns renaming
toyota = toyota.rename(columns = {'Age_08_04': 'Age', 'Mfg_Month': 'Manufacturing_Month', 'Mfg_Year': 'Manufacturing_Year',
                          'Mfr_Guarantee': 'Manufacturer_Guarantee', 'Airco': 'Aircondition', 'Automatic_airco': 'Automatic_Aircondition',
                         'Boardcomputer': 'Board_Computer', 'Power_Steer': 'Powered Steer','Mistlamps': 'Mist_Lamps',
                         'Radio_cassette': 'Radio_Cassette', 'Met_Color': 'Metallic_Color'}
            )
# %% Inspect variables datatypes a
print('The data types of the variables are:\n', toyota.dtypes)

# strip whitespace from categorical variables
toyota['Model'] = toyota['Model'].str.strip()
toyota['Fuel_Type'] = toyota['Fuel_Type'].str.strip()
toyota['Color'] = toyota['Color'].str.strip()
# %%mCheck the range of the numerical variables in our dataset. It helps to identify various potential issues.
toyota_numeric = toyota.select_dtypes(include=['int64', 'float64'])

for col in toyota_numeric:
    print(f'{col}: min = {toyota_numeric[col].min()}, max = {toyota_numeric[col].max()}')

# I found this ad, which includes a car model that is part of the dataset, to help me understand what each variable represents.
# https://autogidas.lt/en/auto-katalogas/toyota/corolla/e12e13-1.4-vvt-i-luna-2006-2007-k63164
# %% Check the number of unique values for each variable. 
# This will help us identify which are the 'real' numerical variables of our dataset.
print(toyota.nunique()) 
# %% Price - Horsepower (HP)
os.makedirs("graphs", exist_ok=True) # Make folder to save graphs

sns.scatterplot(data=toyota, x='HP', y='Price') 
plt.ylim(bottom=0)
plt.title('Price and Horsepower relationship')
plt.savefig("graphs/Prices and Horsepower.png", dpi=300, bbox_inches="tight")
plt.close() # Save RAM space
# %% Price - CC
sns.scatterplot(data=toyota, x='CC', y='Price') 
plt.ylim(bottom=0)
plt.title('Price and CC relationship')
plt.savefig("graphs/Prices and CC.png", dpi=300, bbox_inches="tight")
plt.close()
# %%
print(toyota.loc[toyota['CC'] == 16000])
# %%
print(toyota.loc[toyota['Model'].str.contains('TOYOTA Corolla 1.6 5drs', case = False)]) # It seems only one model of this exists in the dataset
# https://www.autowereld.nl/toyota/corolla/corolla-1-6-met-airco-5-deur-s-39160013/details.html?referrer=https%3A%2F%2Fwww.google.com%2F
# It seems the model has 1600 CC, so I will change the value to 1600 CC

# %%
toyota.loc[toyota['CC'] == 16000, 'CC'] = 1600
sns.scatterplot(data=toyota, x='CC', y='Price') 
plt.ylim(bottom=0)
plt.title('Price and CC relationship (cleaned)')
plt.savefig("graphs/Prices and CC cleaned.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
# Quarterly Tax
sns.scatterplot(data = toyota, x = 'Quarterly_Tax', y = 'Price')
plt.ylim(bottom=0)
plt.title('Price and Quarterly Tax relationship')
plt.savefig("graphs/Prices and Quarterly Tax.png", dpi=300, bbox_inches="tight")
plt.close()
# %% Create a list of the variables that are numerical indeed
toyota_numerical = ['Id', 'Price', 'Age_08_04', 'KM','HP','CC', 'Quarterly_Tax' 'Weight']
# %% Drop irrelevant columns
# Drop columns that are useless in predicting prices
toyota = toyota.drop(columns=['Id', 'Model', 'Cylinders', 'Radio_Cassette', 'Manufacturing_Month', 'Manufacturing_Year']) 
# %% Duplicates check
# Check if we have any duplicates and remove them
print('There are: ', toyota.duplicated().sum(), 'duplicated rows.') # We have 1 duplicated row
toyota = toyota.drop_duplicates()  # Remove duplicate rows
print('After removing the duplicated rows, there are now: ', toyota.duplicated().sum(), 'duplicated rows.')
# %% NA values check
# Check for missing values
print('There are: ',  toyota.isna().sum().sum(), 'missing values in the dataset') # There are no missing values in our dataset
# %% Descriptive statistics of the cleaned dataset
print(toyota.describe())
# %% Save the cleaned dataset
toyota.to_csv('data/toyota_cleaned.csv', index=False)
print(toyota.head().to_string()) # One last check of the cleaned dataset

# %% End of the code