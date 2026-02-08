# %%
# Import useful libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

# %% Directory setup
# Set relative path to the data file
base_dir = os.path.dirname(os.path.abspath(__file__))
toyota_path = os.path.join(base_dir, "data", "toyota_ready_for_modeling.csv")
# %% Import the cleaned dataset and read for modeling dataset
toyota = pd.read_csv(toyota_path) # Insert the dataset as a pandas dataframe
pd.set_option('display.max_columns', None)  # Show all columns of the dataset
print("The dimensions of the cleaned dataset are:", toyota.shape) # Dimensions of the dataset
print(toyota.head()) # First 5 rows of the dataset
# %% Transform Prices to Log Prices
toyota['Log_Price'] = np.log(toyota['Price'])
# %% Train-Test Split
train,test = train_test_split(toyota, test_size=0.2, random_state=42) # 80-20 train-test split
# %% Random Forests
# Random Forest Baseline Model
y_train_log = train["Log_Price"].copy() # Response

X_test_rf_baseline  = test.drop(columns=["Price", "Log_Price"]).copy() # Features
X_train_rf_baseline = train.drop(columns=['Price', 'Log_Price']).copy() 

num_features = ["Age", "KM", "HP", "CC", "Weight","AC", "Airbags_Number"] # List of numerical features

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[], # No need to scale for Random Forests
    remainder="passthrough"
)

# Fit the pipeline to the training data
pipeline_rf_baseline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=500, # Number of trees
        random_state=42, # Reproducibility
        min_samples_leaf= 5, # Minimum number of observations per leaf
        max_features= 'sqrt', # 6 features considered for each split
        n_jobs = -1 # Use full CPU power
    ))
    
])

# Fit the pipeline to the training data
pipeline_rf_baseline.fit(X_train_rf_baseline, y_train_log)
y_test_log = test['Log_Price'].values
y_test_price = test['Price'].values # For plotting 

# Predict on test set
y_pred_log = pipeline_rf_baseline.predict(X_test_rf_baseline)

# 10-fold Cross-Validation to assess model performance
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline_rf_baseline, X_train_rf_baseline, y_train_log, cv = kf, scoring='r2')
print("Cross-validated R² scores for each fold:\n", scores.round(4))
print(f"Mean R² from cross-validation: {np.mean(scores):.4f}")
print(f"Standard deviation of R² scores: {np.std(scores):.4f}")

# Test  performance
test_r2 = pipeline_rf_baseline.score(X_test_rf_baseline, y_test_log)
print('The Test R² is:', round(test_r2, 4))

# Check performance and generalization by comparing CV scores to test R²
gap = np.mean(scores) - test_r2

if gap > np.std(scores):
    print("The model shows signs of overfitting!!!")
elif abs(gap) <= np.std(scores):
    print("The model generalizes well!")
else:
    print("Test performance exceeds CV estimates")

# %% Additional evaluation metrics and plots
price_pred = np.exp(y_pred_log) # Convert log prices back to original prices, rf does not require distributgional assumptions

mae = mean_absolute_error(y_test_price, price_pred)
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Random Forest Baseline Model Plots",
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
    "graphs/Random_Forest_Baseline_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Histogram of tree depths in the baseline Random Forest

# Extract the fitted RandomForestRegressor from the pipeline
rf_model = pipeline_rf_baseline.named_steps["regressor"]

# Get depth of each tree
tree_depths = [estimator.tree_.max_depth for estimator in rf_model.estimators_]

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(tree_depths, bins=15, edgecolor="black")
plt.xlabel("Tree Depth")
plt.ylabel("Number of Trees")
plt.title("Distribution of Tree Depths in Baseline Random Forest Model")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "graphs/Random_Forest_Baseline_Tree_Depths_Histogram.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Mean tree depth
print(f"Mean tree depth: {np.mean(tree_depths):.2f}")
# %% Grid Search CV the Forest to find optimal hyperparameters for Random Forest

# Attention!!!

# This might take a while to run, depending on the machine you use!!!
# On my computer(16GB RAM, 6-core CPU (2.38 GHz)), Grid Search takes around 16 minutes!!!
# TO BE LESS TIME CONSUMING, YOU CAN SET THE FOLLOWING KEY:VALUES to parameter_grid dictionary:
# max_features :30, min_samples_leaf: 3, max_depth: 12, min_samples_split: 5
# These hyperparameter values give the best model.

# Set up the parameter grid for hyperparameter tuning
parameter_grid = {
    "regressor__max_features": ['sqrt', 10, 20, 30, None], # How many features do we consider at each split.
    "regressor__min_samples_leaf": [3, 5, 10], # Every final node (leaf) must have at least X observations.
    "regressor__max_depth": [10, 12, 14, 16, None], # Maximum depth of the tree in the forest chosen based on the histogram of tree depths.
    "regressor__min_samples_split": [2, 5, 10] # Every node must have at least X observations to be considered for splitting.
}

grid = GridSearchCV(
    estimator = pipeline_rf_baseline,
    param_grid = parameter_grid, # The hyperparameters to tune
    scoring = "r2",
    cv = 10, # 10-fold Cross-Validation
    n_jobs =-1, # Use all CPU cores
    return_train_score = True
)

grid.fit(X_train_rf_baseline, y_train_log)

# Best hyperparameters from Grid Search
print(" The best grid parameters are:", grid.best_params_)
print(" The best grid score (R-squared) is:", grid.best_score_)

results = pd.DataFrame(grid.cv_results_)
results[[
    "params",
    "mean_test_score",
    "std_test_score",
    "mean_train_score"
]].sort_values("mean_test_score", ascending=False)

# %% Predict using the tuned RF model

# Store the best estimator from the tuned random forest model
best_rf = grid.best_estimator_

# Predictions
y_pred_log = best_rf.predict(X_test_rf_baseline)

# Test  performance
test_r2 = best_rf.score(X_test_rf_baseline, y_test_log)
print('The Test R² is:', round(test_r2, 4))

best_idx = grid.best_index_ # Index of the best model from Grid Search CV
cv_std = grid.cv_results_["std_test_score"][best_idx]  # Use std of Scores from Grid Search CV 
# Note that you can find all sorts of statistics in grid.cv_results_

cv_mean = grid.best_score_ # Mean CV score of the best model

# Check performance and generalization by comparing best model CV scores to test R²
gap = cv_mean - test_r2

if gap > cv_std:
    print("Possible overfitting to CV folds during tuning.")
elif abs(gap) <= cv_std:
    print("Good generalization after tuning!")
else:
    print("Test performance exceed CV estimates")

# %% Additional evaluation metrics and plots for the tuned Random Forest model

# Convert log prices back to original prices, rf does not require distributional assumptions
price_pred = np.exp(y_pred_log) 

mae = mean_absolute_error(y_test_price, price_pred)
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Random Forest Tuned Model Plots",
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
    "graphs/Random_Forest_Tuned_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Permutation Feature Importances (PIF) of the tuned and fitted random forest

# Compute the PIF
pif = permutation_importance(
    estimator=best_rf, # Best estimator from the tuned random forest model, PIF is model-agnostic.
    X=X_test_rf_baseline, # Unseen test features
    y=y_test_log, # True targets
    scoring="r2", # Evaluate importance based on R² score
    n_repeats=30, # Number of times each feature is permuted
    random_state=42, # Reproducibility                   
    n_jobs=-1 # Use all CPU cores
)

# Store the PIF results in a df
pif_df = pd.DataFrame({
    "Feature": X_test_rf_baseline.columns,
    "Importance_Mean": pif.importances_mean.round(4),
    "Importance_Std": pif.importances_std.round(4)
}).sort_values(by="Importance_Mean", ascending=False) # Sort the table by mean importance descending order

print(pif_df)
unimportant_features = pif_df[pif_df["Importance_Mean"] <= 0]
print("The number of features that are not important are:", len(unimportant_features))
# List of unimportant features to drop
unimportant_features_list = unimportant_features["Feature"].tolist()
print("The unimportant features are:", unimportant_features_list)
#%%
# Bar plot of PIF with error bars that show the st.dev of the feature importance across n_repeats permutations
plt.figure(figsize=(12, 10))
plt.barh(
    pif_df["Feature"],
    pif_df["Importance_Mean"],
    xerr=pif_df["Importance_Std"]
)
plt.xlabel("Decrease in R²  ")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance of Tuned Random Forest Model")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "graphs/Random_Forest_Tuned_Permutation_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Re-train the Random Forest model using only the important features selected by PIF

# Define Features test and training sets after dropping unimportant features
X_test_rf_important  = X_test_rf_baseline.drop(columns=unimportant_features_list).copy() # Features
X_train_rf_important = X_train_rf_baseline.drop(columns=unimportant_features_list).copy()

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[], # No need to scale for Random Forests
    remainder="passthrough"
)
# Fit the pipeline to the training data
pipeline_rf_important = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=500, # Number of trees
        random_state=42, # Reproducibility
        min_samples_leaf= 5, # Minimum number of observations per leaf
        max_features= 'sqrt', # 6 features considered for each split
        n_jobs = -1 # Use full CPU power
    ))
    
])
# Fit the pipeline to the training data
pipeline_rf_important.fit(X_train_rf_important, y_train_log)
y_test_log = test['Log_Price'].values
y_test_price = test['Price'].values # For plotting 

# Predict on test set
y_pred_log = pipeline_rf_important.predict(X_test_rf_important)

# 10-fold Cross-Validation to assess model performance
kf = KFold(n_splits=10, shuffle=True, random_state=42) 
scores = cross_val_score(pipeline_rf_important, X_train_rf_important, y_train_log, cv = kf, scoring='r2')
print("Cross-validated R² scores for each fold:\n", scores.round(4))
print(f"Mean R² from cross-validation: {np.mean(scores):.4f}")
print(f"Standard deviation of R² scores: {np.std(scores):.4f}")

# Test  performance
test_r2 = pipeline_rf_important.score(X_test_rf_important, y_test_log)
print('The Test R² is:', round(test_r2, 4))

# Check performance and generalization by comparing CV scores to test R²
gap = np.mean(scores) - test_r2

if gap > np.std(scores):
    print("The model shows signs of overfitting!!!")
elif abs(gap) <= np.std(scores):
    print("The model generalizes well!")
else:
    print("Test performance exceeds CV estimates")
# %% Additional evaluation metrics and plots
price_pred = np.exp(y_pred_log) # Convert log prices back to original prices, rf does not require distributgional assumptions

mae = mean_absolute_error(y_test_price, price_pred)
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Random Forest Important Features Model Plots",
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
    "graphs/Random_Forest_Important_Features_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Histogram of tree depths in the baseline Random Forest

# Extract the fitted RandomForestRegressor from the pipeline
rf_model = pipeline_rf_important.named_steps["regressor"]

# Get depth of each tree
tree_depths = [estimator.tree_.max_depth for estimator in rf_model.estimators_]

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(tree_depths, bins=15, edgecolor="black")
plt.xlabel("Tree Depth")
plt.ylabel("Number of Trees")
plt.title("Distribution of Tree Depths in Important Features Random Forest Model")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "graphs/Random_Forest_Important_Features_Tree_Depths_Histogram.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Mean tree depth
print(f"Mean tree depth: {np.mean(tree_depths):.2f}")
# %% Re-tune the Random Forest model using only the important features selected by PIF

# Attention!!!

# This might take a while to run, depending on the machine you use!!!
# On my computer(16GB RAM, 6-core CPU (2.38 GHz)), Grid Search takes around 13 minutes!!!
# TO BE LESS TIME CONSUMING, YOU CAN SET THE FOLLOWING KEY:VALUES to parameter_grid dictionary:
# max_features :20, min_samples_leaf: 3, max_depth: 12, min_samples_split: 2
# These hyperparameter values give the best model

# Set up the parameter grid for hyperparameter tuning
parameter_grid = {
    "regressor__max_features": ['sqrt', 10, 20, 30, None], # How many features do we consider at each split.
    "regressor__min_samples_leaf": [3, 5, 10], # Every final node (leaf) must have at least X observations.
    "regressor__max_depth": [10, 12, 14, 16, None], # Maximum depth of the tree in the forest chosen based on the histogram of tree depths.
    "regressor__min_samples_split": [2, 5, 10] # Every node must have at least X observations to be considered for splitting.
}

grid = GridSearchCV(
    estimator = pipeline_rf_important,
    param_grid = parameter_grid, # The hyperparameters to tune
    scoring = "r2",
    cv = 10, # 10-fold Cross-Validation
    n_jobs =-1, # Use all CPU cores
    return_train_score = True
)

grid.fit(X_train_rf_important, y_train_log)

# Best hyperparameters from Grid Search
print(" The best grid parameters are:", grid.best_params_)
print(" The best grid score (R-squared) is:", grid.best_score_)

results = pd.DataFrame(grid.cv_results_)
results[[
    "params",
    "mean_test_score",
    "std_test_score",
    "mean_train_score"
]].sort_values("mean_test_score", ascending=False)
# %%
# Store the best estimator from the tuned random forest model
best_rf = grid.best_estimator_

# Predictions
y_pred_log = best_rf.predict(X_test_rf_important)

# Test  performance
test_r2 = best_rf.score(X_test_rf_important, y_test_log)
print('The Test R² is:', round(test_r2, 4))

best_idx = grid.best_index_ # Index of the best model from Grid Search CV
cv_std = grid.cv_results_["std_test_score"][best_idx]  # Use std of Scores from Grid Search CV 
# Note that you can find all sorts of statistics in grid.cv_results_

cv_mean = grid.best_score_ # Mean CV score of the best model

# Check performance and generalization by comparing best model CV scores to test R²
gap = cv_mean - test_r2

if gap > cv_std:
    print("Possible overfitting to CV folds during tuning.")
elif abs(gap) <= cv_std:
    print("Good generalization after tuning!")
else:
    print("Test performance exceed CV estimates")

# %% Additional evaluation metrics and plots for the re-tuned Random Forest model

# Convert log prices back to original prices, rf does not require distributional assumptions
price_pred = np.exp(y_pred_log) 

mae = mean_absolute_error(y_test_price, price_pred)
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Random Forest Re-Tuned Model Plots",
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
    "graphs/Random_Forest_Re_Tuned_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% Boosting Algorithm using Gradient Boosting Regressor (GBM)

# Response (Defined again for clarity)
y_train_log = train["Log_Price"].copy() 

# Features after dropping unimportant features based on PIF from the tuned Random Forest model
X_test_gbm  = X_test_rf_baseline.drop(columns=unimportant_features_list).copy() 
X_train_gbm = X_train_rf_baseline.drop(columns=unimportant_features_list).copy()

# Define the GBM model
gbm = GradientBoostingRegressor(
        random_state=42,
    loss="squared_error" # Squared error loss function
)

# Set up the hyperparameter grid for tuning GBM
parameter_grid_gbm = [
    {
        "learning_rate": [0.1], # Shrinkage parameter
        "n_estimators": [100, 200], # Number of trees (B)
        "max_depth": [1, 2, 3], # The number of splits in each tree
        "min_samples_leaf": [3, 5] # Minimum number of observations per leaf
    },
    {
        "learning_rate": [0.05],
        "n_estimators": [300, 500],
        "max_depth": [1, 2],
        "min_samples_leaf": [3, 5]
    },
    {
        "learning_rate": [0.01],
        "n_estimators": [800, 1200],
        "max_depth": [1],
        "min_samples_leaf": [5, 10]
    },
    {
        "learning_rate": [0.005, 0.001],
        "n_estimators": [3000, 5000],
        "max_depth": [1],
        "min_samples_leaf": [10]
    }
]

# Grid Search with CV to tune GBM hyperparameters
grid_gbm = GridSearchCV(
    estimator = gbm,
    param_grid = parameter_grid_gbm,
    scoring = "r2",
    cv =10,
    n_jobs=-1,
    return_train_score=True
)

grid_gbm.fit(X_train_gbm, y_train_log)

# Best hyperparameters from Grid Search
print(" The best grid parameters are:", grid_gbm.best_params_)
print(" The best grid score (R-squared) is:", grid_gbm.best_score_)

# Store the results in a dataframe
results_gbm = pd.DataFrame(grid_gbm.cv_results_)
results_gbm[[
    "params",
    "mean_test_score",
    "std_test_score",
    "mean_train_score"
]].sort_values("mean_test_score", ascending=False)
# %% Predict using the tuned GBM model

# Store the best estimator from the tuned GBM model
best_gbm = grid_gbm.best_estimator_

# Predictions
y_pred_log = best_gbm.predict(X_test_gbm)

# Test  performance
test_r2 = best_gbm.score(X_test_gbm, y_test_log)
print('The Test R² is:', round(test_r2, 4))

best_idx = grid_gbm.best_index_ # Index of the best model from Grid Search CV
cv_std = grid_gbm.cv_results_["std_test_score"][best_idx]  # Use std of Scores from Grid Search CV 
# Note that you can find all sorts of statistics in grid_gbm.cv_results_

cv_mean = grid_gbm.best_score_ # Mean CV score of the best model

# Check performance and generalization by comparing best model CV scores to test R²
gap = cv_mean - test_r2

if gap > cv_std:
    print("Possible overfitting to CV folds during tuning.")
elif abs(gap) <= cv_std:
    print("Good generalization after tuning!")
else:
    print("Test performance exceed CV estimates")
# %% Additional evaluation metrics and plots for the tuned GBM model

# Convert log prices back to original prices, rf does not require distributional assumptions
price_pred = np.exp(y_pred_log) 

mae = mean_absolute_error(y_test_price, price_pred)
rmse = np.sqrt(mean_squared_error(y_test_price, price_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Tuned Gradient Boosting Model Plots",
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
    "graphs/Tuned_Gradient_Boosting_Predicted_vs_Actual_and_Relative_Error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# %% End of Code