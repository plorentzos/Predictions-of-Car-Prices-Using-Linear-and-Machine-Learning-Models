# %% Import useful libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# %% Directory setup
# Set relative path to the data file
base_dir = os.path.dirname(os.path.abspath(__file__))
toyota_path = os.path.join(base_dir, "data", "toyota_ready_for_modeling.csv")
# %%
toyota = pd.read_csv(toyota_path) # Insert the dataset as a pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns of the dataset
print("The dimensions of the cleaned dataset are:", toyota.shape) # Dimensions of the dataset
print(toyota.head().to_string()) # First 5 rows of the dataset
# %% Transform Prices to Log Prices
toyota['Log_Price'] = np.log(toyota['Price'])
# %% Simple Linear Regression: Log Prices ~ Age (or Log Y ~ X)
train,test = train_test_split(toyota, test_size=0.2, random_state=42) # 80% train, 20% test split, set seed for reproducibility

# Feature and Response
X_train_simple = train["Age"].values.reshape(-1,1).copy() # Reshape is required by scikit-learn to 2D array, since we regress with only one feature
X_test_simple = test["Age"].values.reshape(-1,1).copy() 

y_train = train['Log_Price'].copy()

# Build the pipeline
pipeline_ols_simple = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Fit the pipeline to the training data
pipeline_ols_simple.fit(X_train_simple, y_train)
  
y_test_log = test['Log_Price'].values.copy() # For evaluation
y_test_price = test['Price'].values.copy() # For plotting 

# Predict the prices for the test set
y_pred_log = pipeline_ols_simple.predict(X_test_simple)

# 10-fold Cross-Validation to assess model performance
# After some experimentation I chose 10 folds to validate, with shuffling since we do not have a time series.
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline_ols_simple, X_train_simple, y_train, cv = kf, scoring='r2')

print("Cross-validated R² scores for each fold:\n", scores.round(4))
print(f"Mean R² from cross-validation: {np.mean(scores):.4f}")
print(f"Standard deviation of Cross-Validated R² scores: {np.std(scores):.4f}")

# Test  performance
test_r2 = pipeline_ols_simple.score(X_test_simple, y_test_log)
print('The Test R² is:', round(test_r2, 4))

# Check performance by comparing CV scores to test R²
gap = np.mean(scores) - test_r2 # Mean CV score - test R²

if gap > np.std(scores):
    print("The model shows signs of overfitting!!!")
elif abs(gap) <= np.std(scores):
    print("The model generalizes well!")
else:
    print("Test performance exceeds training performance")

# %% Additional evaluation metrics and plots

# Transform predictions back to Prices to compute additional metrics and plot predicted vs actual prices
# Y ~ exp(Xβ + 0.5σ²) where σ² is the estimated variance of the residuals

train_pred_log = pipeline_ols_simple.predict(X_train_simple) # Predictions on training set
residuals_log = y_train - train_pred_log # Get residuals on training set
sigma_squared = np.var(residuals_log, ddof=1) # Sample variance with Bessel's correction
price_pred = np.exp(y_pred_log + 0.5 * sigma_squared) # Assume ε ~ N(0, σ²)

mae = mean_absolute_error(y_test_price, price_pred) # Mean absolute error
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred)) # Root mean squared error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Simple Linear Regression Model Plots",
    fontsize=16,
    fontweight="bold"
)
# Predicted vs Actual Prices
axs[0].scatter(y_test_price, price_pred, alpha=0.7)
axs[0].plot(
    [y_test_price.min(), y_test_price.max()],
    [y_test_price.min(), y_test_price.max()],
    'r--',
    lw=2
)
axs[0].set_xlabel('Actual Prices (in €)')
axs[0].set_ylabel('Predicted Prices (in €)')
axs[0].set_title('Predicted vs Actual Prices')
axs[0].set_xlim(left=0)
axs[0].set_ylim(bottom=0)
axs[0].grid(True)

# Relative error calculation
relative_error = np.abs(price_pred - y_test_price) / y_test_price * 100

# Relative Error vs Actual Prices
axs[1].scatter(y_test_price, relative_error, alpha=0.7, color='green')
axs[1].set_xlabel('Actual Prices (in €)')
axs[1].set_ylabel('Relative Error (%)')
axs[1].set_title('Relative Error vs Actual Prices')
axs[1].set_xlim(left=0)
axs[1].set_ylim(0, 100)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(
    "graphs/Simple_LR_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Multiple Linear Regression Model: Log Prices ~ βX + ε, 
# Where, X is a matrix of features, β is a vector of coefficients, and ε is the error term

X_train_multi = train.drop(columns=['Price', 'Log_Price']).copy() # Features
X_test_multi  = test.drop(columns=["Price", "Log_Price"]).copy()

y_train = train["Log_Price"].copy() # Response (defined earlier, just for clarity)

num_features = ["Age", "KM", "HP", "CC", "Weight","AC", "Airbags_Number"] # List of numerical features

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ],
    remainder="passthrough"
)

pipeline_ols_multi = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline to the training data
pipeline_ols_multi.fit(X_train_multi, y_train)

y_test_log = test["Log_Price"].copy() # For evaluation
y_test_price = test["Price"].copy() # For plotting 

# Predict the log prices for the test set
y_pred_log = pipeline_ols_multi.predict(X_test_multi)

# 10-fold Cross-Validation to assess model performance
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline_ols_multi, X_train_multi, y_train, cv = kf, scoring='r2')

print("Cross-validated R² scores for each fold:\n", scores.round(4))
print(f"Mean R² from cross-validation: {np.mean(scores):.4f}")
print(f"Standard deviation of R² scores: {np.std(scores):.4f}")

# Test  performance
test_r2 = pipeline_ols_multi.score(X_test_multi, y_test_log)
print('The Test R² is:', round(test_r2, 4))

# Check performance and generalization by comparing CV scores to test R²
gap = np.mean(scores) - test_r2

if gap > np.std(scores):
    print("The model shows signs of overfitting!!!")
elif abs(gap) <= np.std(scores):
    print("The model generalizes well!")
else:
    print("Test performance exceeds training performance")
# %% Additional evaluation metrics and plots for the multiple linear regression model

# Transform predictions back to Prices to compute additional metrics and plot predicted vs actual prices
# Y ~ exp(Xβ + 0.5σ²) where σ² is the estimated variance of the residuals

train_pred_log = pipeline_ols_multi.predict(X_train_multi) # Predictions on training set
residuals_log = y_train - train_pred_log # Get residuals on training set
sigma_squared = np.var(residuals_log, ddof=1) # Sample variance with Bessel's correction
price_pred = np.exp(y_pred_log + 0.5 * sigma_squared) # Assume ε ~ N(0, σ²)

mae = mean_absolute_error(y_test_price, price_pred) # Mean absolute error
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred)) # Root mean squared error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Multiple Linear Regression Model Plots",
    fontsize=16,
    fontweight="bold"
)
# Predicted vs Actual Prices
axs[0].scatter(y_test_price, price_pred, alpha=0.7)
axs[0].plot(
    [y_test_price.min(), y_test_price.max()],
    [y_test_price.min(), y_test_price.max()],
    'r--',
    lw=2
)
axs[0].set_xlabel('Actual Prices (in €)')
axs[0].set_ylabel('Predicted Prices (in €)')
axs[0].set_title('Predicted vs Actual Prices')
axs[0].set_xlim(left=0)
axs[0].set_ylim(bottom=0)
axs[0].grid(True)

# Relative error calculation
relative_error = np.abs(price_pred - y_test_price) / y_test_price * 100

# Relative Error vs Actual Prices
axs[1].scatter(y_test_price, relative_error, alpha=0.7, color='green')
axs[1].set_xlabel('Actual Prices (in €)')
axs[1].set_ylabel('Relative Error (%)')
axs[1].set_title('Relative Error vs Actual Prices')
axs[1].set_xlim(left=0)
axs[1].set_ylim(0, 100)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(
    "graphs/Multiple_LR_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Lasso Regression with Cross-Validation for feature selection and sparsity
X_train_lasso = train.drop(columns=['Price', 'Log_Price']).copy() # Features
X_test_lasso  = test.drop(columns=["Price", "Log_Price"]).copy()

y_train = train["Log_Price"].copy() # Response (defined earlier, just for clarity)
num_features = ["Age", "KM", "HP", "CC", "Weight","AC", "Airbags_Number"] # List of numerical features

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ],
    remainder="passthrough"
)

pipeline_lasso = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LassoCV(
        alphas= np.logspace(start = -4, stop =1, num =100), # Hyperparameter tuning: Try 100 alphas from 10^-4 to 10^1 
        random_state=42, 
        max_iter=10000 # Maximum number of iterations
    ))
])

# Fit the lasso pipeline to the training data
pipeline_lasso.fit(X_train_lasso, y_train)
 
# Predict
y_pred = pipeline_lasso.predict(X_test_lasso)

# Get feature names after preprocessing
feature_names = pipeline_lasso.named_steps['preprocessor'].get_feature_names_out()

# Get coefficients
coefficients = pipeline_lasso.named_steps['regressor'].coef_

# Create a feature importance table
lasso_results = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef = pipeline_lasso.named_steps['regressor'].coef_ # Get the coefs
selected_features = feature_names[coef != 0] # Select features with non-zero coefs

# Which features were selected?
print("Selected features:")
for f in selected_features:
    print(f)

# Number of selected features
print("\nThere were ",len(selected_features),'features selected by Lasso Regression.')

# Features that were forced to zero
dropped_features = feature_names[coef == 0]

print("\nDropped features:")
for f in dropped_features:
    print(f)

# Number of dropped features
print("\nThere were ",len(dropped_features),'features dropped by Lasso Regression.')
# %%
# Rescale training data explicitly for plots and analysis through accessing the preprocessor step of the pipeline_lasso
X_train_transformed = pipeline_lasso.named_steps['preprocessor'].transform(X_train_lasso)

# Hyperparameter tuning for plotting the Lasso path
alphas = pipeline_lasso.named_steps['regressor'].alphas_ # all alphas tested during CV
best_alpha = pipeline_lasso.named_steps['regressor'].alpha_ # best alpha chosen by CV

alphas, coefs, _ = lasso_path(
    X_train_transformed,
    y_train,
    alphas=alphas
)

# For labeling the plot
clean_feature_names = [f.split("__")[-1] for f in feature_names]

plt.figure(figsize=(12, 7))
colors = plt.cm.tab20(np.linspace(0, 1, coefs.shape[0]))
for i in range(coefs.shape[0]):
    plt.plot(
        alphas,
        coefs[i, :],
        color=colors[i],
        alpha=0.6,
        linewidth=1.0,
        label=clean_feature_names[i]
    )

ax = plt.gca()
plt.axvline(best_alpha, linestyle='--', color='black')
plt.axvline(
    best_alpha,
    linestyle='--',
    color='black',
    linewidth=1.5,
    label=f"Best alpha = {best_alpha:.2e}"
)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient')
plt.title('Lasso Regularization Path')
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0,
    fontsize=9,
    ncol=2,
    title="Features"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "graphs/Lasso Regularization Path_Linear Modeling.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Re-run reduced OLS model with the selected features from Lasso
X_train_reduced = train.drop(columns=['Price', 'Log_Price', 'ABS', 'Board_Computer', 'Central_Lock',
                                    'Power_Steering', 'Sport_Model', 'Metallic_Rim', 'Long_Guarantee',
                                    'Color_Beige', 'Color_Blue', 'Color_Violet', 'Color_Yellow']).copy() # Features
X_test_reduced = test.drop(columns=["Price", "Log_Price", "ABS", "Board_Computer", "Central_Lock",
                                  "Power_Steering", "Sport_Model", "Metallic_Rim", "Long_Guarantee",
                                  "Color_Beige", "Color_Blue", "Color_Violet", "Color_Yellow"]).copy()

y_train = train["Log_Price"].copy() # Response (defined earlier, just for clarity)

num_features = ["Age", "KM", "HP", "CC", "Weight", 'AC', 'Airbags_Number'] # List of numerical features

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ],
    remainder="passthrough"
)

pipeline_reduced_ols_after_lasso = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline to the training data
pipeline_reduced_ols_after_lasso.fit(X_train_reduced, y_train)

y_test_log = test['Log_Price'].copy()
y_test_price = test['Price'].copy() # For plotting 

# Predict the log prices for the test set
y_pred_log = pipeline_reduced_ols_after_lasso.predict(X_test_reduced)

# 10-fold Cross-Validation to assess model performance
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline_reduced_ols_after_lasso, X_train_reduced, y_train, cv = kf, scoring='r2')
print("Cross-validated R² scores for each fold:\n", scores.round(4))
print(f"Mean R² from cross-validation: {np.mean(scores):.4f}")
print(f"Standard deviation of R² scores: {np.std(scores):.4f}")

# Test  performance
test_r2 = pipeline_reduced_ols_after_lasso.score(X_test_reduced, y_test_log)
print('The Test R² is:', round(test_r2, 4))

# Check performance and generalization by comparing CV scores to test R²
gap = np.mean(scores) - test_r2

if gap > np.std(scores):
    print("The model shows signs of overfitting!!!")
elif abs(gap) <= np.std(scores):
    print("The model generalizes well!")
else:
    print("Test performance exceeds training performance")
# %% Additional evaluation metrics and plots for the multiple linear regression model

# Transform predictions back to Prices to compute additional metrics and plot predicted vs actual prices
# Y ~ exp(Xβ + 0.5σ²) where σ² is the estimated variance of the residuals

train_pred_log = pipeline_reduced_ols_after_lasso.predict(X_train_reduced) # Predictions on training set
residuals_log = y_train - train_pred_log # Get residuals on training set
sigma_squared = np.var(residuals_log, ddof=1) # Sample variance with Bessel's correction
price_pred = np.exp(y_pred_log + 0.5 * sigma_squared) # Assume ε ~ N(0, σ²)

mae = mean_absolute_error(y_test_price, price_pred) # mean absolute error
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred)) # root mean squared error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Multiple Linear Regression Post-Lasso Reduced Model Plots",
    fontsize=16,
    fontweight="bold"
)
# Predicted vs Actual Prices
axs[0].scatter(y_test_price, price_pred, alpha=0.7)
axs[0].plot(
    [y_test_price.min(), y_test_price.max()],
    [y_test_price.min(), y_test_price.max()],
    'r--',
    lw=2
)
axs[0].set_xlabel('Actual Prices (in €)')
axs[0].set_ylabel('Predicted Prices (in €)')
axs[0].set_title('Predicted vs Actual Prices')
axs[0].set_xlim(left=0)
axs[0].set_ylim(bottom=0)
axs[0].grid(True)

# Relative error calculation
relative_error = np.abs(price_pred - y_test_price) / y_test_price * 100

# Relative Error vs Actual Prices
axs[1].scatter(y_test_price, relative_error, alpha=0.7, color='green')
axs[1].set_xlabel('Actual Prices (in €)')
axs[1].set_ylabel('Relative Error (%)')
axs[1].set_title('Relative Error vs Actual Prices')
axs[1].set_xlim(left=0)
axs[1].set_ylim(0, 100)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(
    "graphs/Reduced_PostLasso_LR_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Elastic Net Regularization with Cross-Validation

# Note: We go back to full feature set used in original multiple linear regression.

# Let's see if Elastic Net can improve our model over Lasso by combining L1 and L2 penalties
X_train_en = train.drop(columns=['Price', 'Log_Price']).copy() # Features
X_test_en  = test.drop(columns=["Price", "Log_Price"]).copy()

y_train = train["Log_Price"].copy() # Response (defined earlier, just for clarity)
num_features = ["Age", "KM", "HP", "CC", "Weight","AC", "Airbags_Number"] # List of numerical features

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ],
    remainder="passthrough"
)
# Elastic Net with Cross-Validation
pipeline_elasticnet = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", ElasticNetCV(
        l1_ratio=[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1],  # Avoid pure ridge (l1_ratio=0)
        alphas= np.logspace(start =-4, stop = 1, num =100),           
        cv=10,
        random_state=42,
        max_iter=10000
    ))
])

# Fit the pipeline to the training data
pipeline_elasticnet.fit(X_train_en, y_train)

# Predict on test set
y_pred_elasticnet = pipeline_elasticnet.predict(X_test_en)
best_alpha_enet = pipeline_elasticnet.named_steps['regressor'].alpha_
best_l1_ratio = pipeline_elasticnet.named_steps['regressor'].l1_ratio_

# What is the best alpha for the elastic net
print(f"Best alpha: {best_alpha_enet:.5f}")
print(f"Best l1_ratio: {best_l1_ratio}")
print('Since l1_ratio is almost 0, the model leans more towards using a Ridge Regression for prediction')
# %% Ridge Regression with Cross-Validation

# Get again the full dataset and process it like we did with original full multiple linear regression
X_train_ridge = train.drop(columns=['Price', 'Log_Price']).copy() # Features
X_test_ridge  = test.drop(columns=["Price", "Log_Price"]).copy()

y_train = train["Log_Price"].copy() # Response (defined earlier, just for clarity)
num_features = ["Age", "KM", "HP", "CC", "Weight","AC", "Airbags_Number"] # List of numerical features

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ],
    remainder="passthrough"
)

# Fit the pipeline to the training data
pipeline_ridge = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RidgeCV( # CV to find the best alpha for Ridge Regression
        alphas=np.logspace(start =-4, stop = 1, num = 100),  
        cv=10,
        scoring="r2"
    ))
])

# Fit the pipeline to the training data
pipeline_ridge.fit(X_train_ridge, y_train)

y_test_log = test['Log_Price'].values
y_test_price = test['Price'].values # For plotting 

# Predict on test set
y_pred_log = pipeline_ridge.predict(X_test_ridge)

# 10-fold Cross-Validation to assess model performance
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline_ridge, X_train_ridge, y_train, cv = kf, scoring='r2')
print("Cross-validated R² scores for each fold:\n", scores.round(4))
print(f"Mean R² from cross-validation: {np.mean(scores):.4f}")
print(f"Standard deviation of R² scores: {np.std(scores):.4f}")

# Test  performance
test_r2 = pipeline_ridge.score(X_test_ridge, y_test_log)
print('The Test R² is:', round(test_r2, 4))

# Check tuning performance and generalization by comparing CV scores to test R²
gap = np.mean(scores) - test_r2

if gap > np.std(scores):
    print("Possible overfitting to CV folds during tuning.")
elif abs(gap) <= np.std(scores):
    print("Good generalization after tuning!")
else:
    print("Test performance exceed CV estimates")
# %% Predicted vs Actual Prices and Relative Error Plots for Ridge Regression

train_pred_log = pipeline_ridge.predict(X_train_ridge) # Predictions on training set
residuals_log = y_train - train_pred_log # Get residuals on training set
sigma_squared = np.var(residuals_log, ddof=1) # Sample variance with Bessel's correction
price_pred = np.exp(y_pred_log + 0.5 * sigma_squared) # Assume ε ~ N(0, σ²)

mae = mean_absolute_error(y_test_price, price_pred) # Mean absolute error
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred)) # Root mean squared error

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Ridge Regression Model Plots",
    fontsize=16,
    fontweight="bold"
)
# Predicted vs Actual Prices
axs[0].scatter(y_test_price, price_pred, alpha=0.7)
axs[0].plot(
    [y_test_price.min(), y_test_price.max()],
    [y_test_price.min(), y_test_price.max()],
    'r--',
    lw=2
)
axs[0].set_xlabel('Actual Prices (in €)')
axs[0].set_ylabel('Predicted Prices (in €)')
axs[0].set_title('Predicted vs Actual Prices')
axs[0].set_xlim(left=0)
axs[0].set_ylim(bottom=0)
axs[0].grid(True)

# Relative error calculation
relative_error = np.abs(price_pred - y_test_price) / y_test_price * 100

# Relative Error vs Actual Prices
axs[1].scatter(y_test_price, relative_error, alpha=0.7, color='green')
axs[1].set_xlabel('Actual Prices (in €)')
axs[1].set_ylabel('Relative Error (%)')
axs[1].set_title('Relative Error vs Actual Prices')
axs[1].set_xlim(left=0)
axs[1].set_ylim(0, 100)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(
    "graphs/Ridge_Regression_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
#%% End of Code