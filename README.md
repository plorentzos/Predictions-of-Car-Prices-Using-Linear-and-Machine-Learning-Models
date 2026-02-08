# Predictions-of-Car-Prices-Using-Linear-and-Machine-Learning-Models

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Tools and Technologies](#tools-and-technologies)
- [Analysis Workflow](#analysis-workflow)
- [Modeling Assumptions](#modeling-assumptions)
- [Insights](#insights)
- [How to Run](#how-to-run)

## Project Overview

For this project, I use a dataset that contains information about 1436 Toyota Corolla Cars in order to predict their prices.

The analysis includes data cleaning, data visualization, feature engineering, linear modeling, nonlinear modeling and a wide range of statistical and ML algorithms to gain insights about the behaviour of the employed models.

The analysis is performed in Python. 

## Data
A single dataset that is used for the analysis. The dataset can be found on: https://www.kaggle.com/datasets/klkwak/toyotacorollacsv/data

The dataset includes information regarding the prices and multiple features of 1436 Toyota Corolla Cars.

Here is the list of the variables that can be found in the raw dataset:

**Response Variable:**
- `Price` – Selling price of the car (in €)

**Features:**
- `Id` - The row index
- `Model` - The model of the car
- `Age_08_04` - The age of the car in months
- `Mfg_Month` - The month in a given year the car was manufactured
- `Mfg_Year` - The year the car was manufactured
- `KM` -  The mileage of the car (measured in Kilometers)
- `Fuel_Type` - The fuel type of the car (Petrol, Diesel, or CNG)
- `HP` - The horsepower of the car
- `Met_Color` - Binary indicator that shows whether the car has a metallic color (1 = Metallic, 0 = Not Metallic)
- `Color` - The color of the car (e.g. Blue, Red, Green etc.)
- `Automatic` - Binary indicator of automatic transmission of the car (1 = automatic, 0 = manual)
- `CC` - The engine size of the car
- `Doors` – The number of doors the car has
- `Cylinders` – The number of cylinders the car has
- `Gears` - The number of gears the car has
- `Quarterly_Tax` - Quarterly Tax of the car (in €)
- `Weight` - The weight of the car (in kilograms)
- `Mfr_Guarantee` - Binary indicator of whether the car has a manufacturing guarantee (1 = yes, 0 = no)
- `BOVAG_Guarantee` - Binary indicator of whether the car has a BOVAG guarantee (1 = yes, 0 = no)
- `Guarantee_Period` - The number of months a car has in guarantee when purchased
- `ABS` - Binary indicator of ABS (1 = yes, 0 = no)
- `Airbag_1` - Binary indicator of whether the car has an airbag (1 = yes, 0 = no)
- `Airbag_2` - Binary indicator of whether the car has a second airbag (1 = yes, 0 = no)
- `Airco` - Binary indicator of aircondition (1 = yes, 0 = no)
- `Automatic_airco` - Binary indicator of automatic aircondition (1 = yes, 0 = no)
- `Boardcomputer` - Binary indicator of board computer in the car (1 = yes, 0 = no)
- `CD_Player` – Binary indicator of CD player in the car (1 = yes, 0 = no)
- `Central_Lock` – Binary indicator of central locking system (1 = yes, 0 = no)
- `Powered_Windows` – Binary indicator of powered windows (1 = yes, 0 = no)
- `Power_Steering` – Binary indicator of power steering (1 = yes, 0 = no)
- `Radio` – Binary indicator of radio in the car (1 = yes, 0 = no)
- `Mistlamps` – Binary indicator of mist lamps (1 = yes, 0 = no)
- `Sport_Model` – Binary indicator of sport model version (1 = yes, 0 = no)
- `Backseat_Divider` – Binary indicator of backseat divider (1 = yes, 0 = no)
- `Metallic_Rim` – Binary indicator of metallic rims (1 = yes, 0 = no)
- `Radio_cassette` – Binary indicator of radio cassette player (1 = yes, 0 = no)
- `Parking_Assistant` – Binary indicator of parking assistant system (1 = yes, 0 = no)
- `Tow_Bar` – Binary indicator of tow bar (1 = yes, 0 = no)


Note that datasets that are generated during the analysis can be found in data_created folder in the data folder.
Note that graphs generated during the analysis can be found in the graphs folder.
Note that you can always delete the data_created and graphs folder. They will get generated again when you run the .py files.

## Tools and Technologies
* Python (Libraries: os, Pandas, NumPy, Seaborn, Matplotlib, sklearn)

## Analysis Workflow

### Cleaning the data 
The following relate to toyota_cleaned.py script:
* Loaded the raw dataset, performed initial data inspection and renamed columns to improve readibility and consistency.
* Inspected variables data types, performed whitespace trimming, separated numerical and categorical variables.
* Plotted the response variable against features that take a significant amount of unique values (i.e. non binary variables) to find out outliers, invalid data and their relationships with prices.
* Dropped variables that are not useful for the analysis. Checked for missing values and duplicate rows.

### Inspecting the Response Variable (Car Prices)
The following relate to prices_insights.py:
* Imported the cleaned dataset and inspected it for one more time to be sure it is ok to go.
* Made a histogram and kde plot to view the distribution of prices.
* Assessed the relationship of price with the seemingly most important features via scatterplots and simple linear regressions.
* Plotted the residuals of the Price ~ Age linear regression to check for heteroskedasticity. Log transformed prices to eliminate heteroskedastic errors.

### Feature Engineering:
The following relate to feature_engineering.py:
* Performed feature engineering on multiple features by combining them, changing their structure or dropping them.
* Plotted a correlation heatmap of the full cleaned and engineered dataset to see the association between variables and further assess potential issues.
* Constructed a Variance Inflation Factor (VIF) to adress any multicollinearity concerns that I was not able to see with plots and ispection.
* The dataset is now ready for modeling!

### Linear Modeling:
The following relate to linear_modeling.py:

* Transformed prices to log prices to use as a response variable based on previous insights. 
* Constructed all the models (Simple OLS, Multiple OLS, Reduced OLS, Ridge Regression) based on the following workflow:
Splitted the dataset into training set and test set (80-20 split). Created a pipeline to transform and scale features. Fitted the pipeline to the training data. Predicted using the features test set. Performed a 10-Fold Cross Validation on the training sets to assess model performance using R-squared as the metric. Made a decision rule to assess if the model overfits. Transformed log prices back to prices in order to plot actual vs predicted prices and relative errors. Calculated Mean Absolute Error and Root Mean Squared Error as additional metrics.
* Performed regularization via Lasso(L1), Ridge(L2) and ElasticNet. Plotted Lasso paths.

### Nonlinear Modeling:
The following relate to nonlinear_modeling.py:

* Performed similar workflow as in linear modeling section bullets 1 and 2.
* Tuned the hyperparameters  for all of the  Machine Learning models (Random Forests, Gradient Boosting). (I provide the optimal hyperparameters so you do not waste time running all iterations when doing grid search, read the code comments for further information!!!)
* Performed Permutation Feature Importances (PIF) to optimize further the models by dropping features that do not contribute to goodness of fit.
* Re-fitted and re-tuned the models Post-PIF.

Note that additional comments regarding the code and the analysis can be found in the .py files.

## Modeling Assumptions
* The dataset's size is large enough to train/test split and apply the various statistical methods and models.
* Residual diagnostics indicated heteroskedasticity in the original simple regression model. This subsequently stays for all of the models. Hence prices are transformed to log prices to eliminate most of the heteroskedasticity.
* Used Ordinary Least Squares methods initially to understand what is happening with the dataset and gain insights since the models are interpretable. For OLS methods, Assume the errors terms are normal distributed with 0 mean and finite variance.
* Use ML algorithms since we do predictions and not causal inference. The dataset is sizeable enough for these methods.
* Measure the model performance on how high the R-Squared is but also rely on MAE and RMSE as additional metrics to get a sense how good the predictions are related to prices in euros, outliers in the dataset and penalties imposed on the error terms.

 
## Insights
I present below a table thats contains the evaluation metrics for most of the models used for the analysis. You can find by running the code for the models I do not include here, but I find it uneccessary to present here their metrics. Overall, I choose to not include in the table the simple linear regression results and the not-tuned(baseline) ML models results.

| Model | R² | RMSE (€) | MAE (€) |
|------|----:|---------:|--------:|
| Multiple OLS| 0.8582 | 1,172.58 | 834.30 |
| Reduced Post-Lasso OLS | 0.8586 | 1,169.86 | 828.40 |
| RidgeCV | 0.8572 | 1,184.11 | 831.42 |
| RF Tuned | 0.8774 | 1,085.56 | 766.17 |
| RF Post-PIF Tuned | 0.8783 | 1,083.74 | 770.10 |
| GBM Post-PIF Tuned | 0.8790 | 1,051.98 | 757.09 |

OLS Stands for Ordinary Leaset Squared, RidgeCV stands for Cross Validated Ridge Regression, RF stands for Random Forest, GBM stands for Gradient Boosting Machine.

All metrics confirm the robustness of the modeling assumptions. 
Hence, based on performance all metrics so that GBM (Gradient Boosting Machine) algorithm has the best performance

## How to Run 
1. Make sure you have at least Python version 3.9+ installed in your personal computer.
2. Clone the repository.
3. Open the repository using your preferred IDE(e.g. VS Code).
4. Navigate to the project folder where you saved the repo:
   cd Predictions-of-Car-Prices-Using-Linear-and-Machine-Learning-Models
5. Create a virtual environment using uv: python -m uv venv
6. Activate the virtual environment according to the operating system you use(e.g. Windows, Linux, macOS).
7. Install the necessary libraries in the activated virtual environment:
   python -m pip install -r requirements.txt
8. Run the python scripts in this order: toyota_cleaned.py -> prices_insights.py -> feature_engineering.py -> linear_modeling.py -> nonlinear_modeling.py
