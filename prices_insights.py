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
# %% Import the cleaned dataset
toyota = pd.read_csv(toyota_path) # Insert the dataset as a pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns of the dataset
print("The dimensions of the cleaned dataset are:", toyota.shape) # Dimensions of the dataset
print(toyota.head().to_string()) # First 5 rows of the dataset
# %% Distribution of prices
bin_width = 1000  # Each bin will represent a range of 1000€
bins = np.arange(start = 0, stop = toyota['Price'].max() + bin_width, step = bin_width)
                 
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.histplot(toyota['Price'], bins = bins, color = 'green', edgecolor = 'black', alpha = 0.7, ax = ax1)
ax1.set_ylabel('Number of Cars', fontsize=12)
ax1.set_xlabel('Price (€)', fontsize=12 )

ax2 = ax1.twinx() # Create a twin axis

sns.kdeplot(toyota['Price'], color = 'red', linewidth = 1.5, alpha = 0.6, ax = ax2)
ax2.set_ylabel('Density', fontsize = 12)
plt.title('Car Prices Distribution')
ax1.grid(axis = 'y', alpha = 0.5)

plt.savefig("graphs/Car Prices Distribution.png", dpi=300, bbox_inches="tight")
plt.close() # Save RAM space
# %% Prices - Age
# Graph of Prices and Age
plt.figure(figsize = (10,6))

sns.regplot(x=toyota['Age'], y=toyota['Price'], 
            scatter_kws={'s': 10, 'color': 'green'}, 
            line_kws={'color': 'red'})

plt.ylim(bottom=0)
plt.title('Car Price per Age')
plt.xlabel('Age of the car (in months)')
plt.ylabel('Price (€)')

plt.savefig("graphs/Prices and Age.png", dpi=300, bbox_inches="tight")
plt.close()

# Simple linear regression between Prices and Age to quantify the relationship
X = toyota['Age'].values # independent variable
Y = toyota['Price'].values # dependent variable
A = np.vstack([X, np.ones(len(X))]).T # transpose the matrix to have two columns: one for X and one for the intercept term (1s)
XTX = A.T @ A 
XTY = A.T @ Y 
coefs = np.linalg.solve(XTX,XTY) 
Y_hat_raw = A @ coefs
resid_raw = Y - Y_hat_raw

print('The slope of the regression line is:', coefs[0].round(2)) 
print('The intercept of the regression line is:', coefs[1].round(2)) 

# Plot the reisduals to check for heteroscedasticity
plt.figure(figsize=(10,6))
plt.scatter(Y_hat_raw, resid_raw, alpha=0.5)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("Fitted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot (Raw Prices)")

plt.savefig("graphs/Fitted Raw Prices and Age Residuals plot.png", dpi=300, bbox_inches="tight")
plt.close()
# %% Simple linear regression between log(Prices) and Age
Y = np.log(toyota['Price'].values)   # <-- log transform
A = np.vstack([X, np.ones(len(X))]).T
XTX = A.T @ A
XTY = A.T @ Y
coefs_log = np.linalg.solve(XTX, XTY)
Y_hat_log = A @ coefs_log
resid_log = Y - Y_hat_log

print('The slope of the regression line is:', coefs_log[0].round(4))
print('The intercept of the regression line is:', coefs_log[1].round(4))

plt.figure(figsize=(10,6))
plt.scatter(Y_hat_log, resid_log, alpha=0.4, s=15)
plt.axhline(0, linestyle='--', color='black')
plt.xlabel("Fitted Log Price")
plt.ylabel("Residuals")
plt.title("Residuals Plot (Log Prices)")

plt.savefig("graphs/Fitted Log Prices and Age Residuals plot.png", dpi=300, bbox_inches="tight")
plt.close()

# Hence, according to the distribution of prices and the residual plots, log-transforming the prices is a better approach for further analysis.
# %% Prices - Kilometers
# Graph of Prices and Kilometers
plt.figure(figsize = (10,6))

sns.regplot(x=toyota['KM'], y=toyota['Price'], 
            scatter_kws={'s': 10, 'color': 'green'}, 
            line_kws={'color': 'red'})

plt.title('Car Prices per Kilometer')
plt.xlabel('Mileage of the car (in Kilometers)')
plt.ylabel('Price (€)')

plt.savefig("graphs/Prices and Kilometers.png", dpi=300, bbox_inches="tight")
plt.close()

# As the kilometers a car has been driven increase, prices drop. Again, we have a negative relationship between the variables.

# Simple linear regression between Prices and Kilometers
X = toyota['KM'].values # independent variable
Y = toyota['Price'].values # dependent variable
A = np.vstack([X, np.ones(len(X))]).T # transpose the matrix to have two columns: one for X and one for the intercept term (1s)
XTX = A.T @ A 
XTY = A.T @ Y 
coefs = np.linalg.solve(XTX,XTY) 

print('The slope of the regression line is:', coefs[0].round(2)) 
print('The intercept of the regression line is:', coefs[1].round(2)) 
# For every 1,000 kilometers driven the price of a car drops by 50€, ceteris paribus. 
# Also, the average car in the dataset has been driven for approximately 14,479 kilometers.
# %% Prices - ABS (Anti-lock Braking System)
Q1 = toyota['Price'].quantile(0.25) 
Q3 = toyota['Price'].quantile(0.75)
IQR = Q3 - Q1 
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_prices = toyota[(toyota['Price'] >= lower_bound) & (toyota['Price'] <= upper_bound)]

plt.figure(figsize=(12, 6))
sns.boxplot(x = filtered_prices['ABS'], y = filtered_prices['Price'], showfliers = True, fliersize= 3)

plt.ylim(bottom=0)
plt.title('Box plot of Car Prices by ABS')
plt.xlabel('ABS')
plt.ylabel('Price (€)')
plt.grid(axis = 'y', alpha = 0.75)

plt.savefig("graphs/Prices and ABS.png", dpi=300, bbox_inches="tight")
plt.close()

print("On average, cars with ABS are sold at a higher price than cars without ABS.")
print('Original number of rows:', toyota.shape[0])
print('Filtered number of rows:', filtered_prices.shape[0])
print('There were', toyota.shape[0] - filtered_prices.shape[0], 'outliers removed.')

# %%  End of the Code