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

For this project, a dataset that contains information about 1436 Toyota Corolla Cars is used. 

The aim of the project is to find the best model to predict car prices.

The analysis includes data cleaning, data visualization, feature engineering, statistical modeling, linear modeling and machine learning in order make accurate predictions of car prices.

Multiple models are constructed based on standard textbook statistical and mathematical techniques. 

Instead of just presenting "the single best model", the analysis presents findings from multiple traditional and state-of-the-art models that serve as robustness checks for the most optimal model. 

Essentially, as it can been seen in the code the modeling logic is to start simple and build stepwise better (and perhaps more complicated) methods in order to arrive to the optimal result. This logic gives the reader not only a better sense in the coding workflow but also depicts the intuition behind the choices taken during the project.

The analysis is performed in Python programming language. The IDE used is Visual Studio Code. 

## Data
A single dataset  is used for the analysis. The dataset can be found on: https://www.kaggle.com/datasets/klkwak/toyotacorollacsv/data

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
- `Weight` - The weight of the car (in Kilograms)
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
Note that you can always delete the data_created and graphs folder. They will get generated again when you run the .py files in the folder you save the cloned repository.

## Tools and Technologies
* Python (Libraries: os, Pandas, NumPy, Seaborn, Matplotlib, sklearn)

## Project Workflow

### Cleaning the data 
The following relate to toyota_cleaned.py script:

* Loaded the raw dataset, performed initial data inspection and renamed columns to improve readibility and consistency.
* Inspected variables data types, performed whitespace trimming, classified the datatypes of each feature(e.g. numerical feature etc.)
* Plotted the response variable against features that take a significant amount of unique values (i.e. non binary variables) to find out outliers, invalid data and their relationships with prices.
* Dropped variables that are not useful for the analysis. Checked for missing values and duplicate rows.

### Inspecting the Response Variable (Car Prices)
The following relate to prices_insights.py:

* Imported the cleaned dataset and inspected it again.
* Made a histogram and kde plot to view the distribution of prices.
* Assessed the relationship of price with the seemingly most important features via scatterplots and simple linear regressions.
* Plotted the residuals of the Price ~ Age linear regression to check for heteroskedasticity. Log transformed prices to eliminate heteroskedastic errors.

### Feature Engineering:
The following relate to feature_engineering.py:

* Performed feature engineering on multiple features by combining them, changing their structure or dropping them.
* Plotted a correlation heatmap of the full cleaned and engineered dataset to see the association between variables and further assess potential issues.
* Constructed a Variance Inflation Factor (VIF) to adress any multicollinearity concerns that could not be salient from the data inspection and correlation heatmap.
* The dataset is now ready for modeling!

### Linear Modeling:
The following relate to linear_modeling.py:

* Transformed prices to log prices to use as a response variable based on previous insights. 
* Constructed all the models (Simple OLS, Multiple OLS, Reduced OLS, Ridge Regression) based on the following workflow:
Splitted the dataset into training set and test set (80-20 split). Created a pipeline to transform and scale features. Fitted the pipeline to the training data. Predicted using the features test set. Performed a 10-Fold Cross Validation on the training sets to assess model performance using R-squared as the metric. Made a robust decision rule to assess if the model overfits, if the test score is significantly higher than the training score or if everything is ok. Transformed log prices back to prices in order to plot actual vs predicted prices and relative errors. Calculated Mean Absolute Error and Root Mean Squared Error as additional metrics.
* Performed regularization via Lasso(L1), Ridge(L2) and ElasticNet. Plotted Lasso paths.

### Nonlinear Modeling:
The following relate to nonlinear_modeling.py:

* Performed similar workflow as in linear modeling section bullets 1 and 2.
* Tuned the hyperparameters  for all of the  Machine Learning models (Random Forests, Gradient Boosting). (!!!I provide the optimal hyperparameters so you do not waste time running all iterations when doing grid search, read the code comments for further information!!!)
* Performed Permutation Feature Importances (PIF) to optimize further the models by selecting a subset of features deemed as important.
* Re-fitted and re-tuned the models post PIF.

Note that additional comments regarding the code and the analysis can be found in the .py files.

## Assumptions
* The dataset's size is large enough to train/test split and apply the various statistical methods and models.
* Splitting train = 80%, test = 20% is textbook standard. Did not choose to split to train, test and validations sets. Used CV to validate the training sets.
* Residual diagnostics indicated heteroskedasticity in the original simple regression model. This subsequently stays for all of the models. Hence, prices are transformed to log prices to eliminate most of the heteroskedasticity.
* A number of features that are in dataset introduce noise to the models since they either add irrelevant information. Hence, they are dropped before modeling.
* Used Ordinary Least Squares methods initially to understand what is happening with the dataset and gain insights since OLS is easy to interpret. For OLS methods, the error terms are assumed to be normally distributed with 0 mean and finite variance.
* Instead of relying to more advanced econometric methods,  ML algorithms were employed since we do predictions and not causal inference. The dataset is sizeable enough for these methods.
* As a primary measure to evaluate the predictions, R-squared is used. However, MAE is the primary metric of interest.

 
## Insights
A table is presented below thats contains the evaluation metrics for most of the models used for the analysis. There are multiple models in the code that are not shown in the table. Those models are considered as prerequisite steps to find the optimal model using each different technique. Specifically, the simple linear regression results and the not-tuned(baseline) ML models results are not presented for simplicity reasons.

| Model | MAE (€) | RMSE (€) | R² |
|------|--------:|---------:|---:|
| Multiple OLS | 834.30 | 1,172.58 | 0.8582 |
| Reduced Post-Lasso OLS | 828.40 | 1,169.86 | 0.8586 |
| RidgeCV | 831.42 | 1,184.11 | 0.8572 |
| RF Tuned | 766.17 | 1,085.56 | 0.8774 |
| RF Post-PIF Tuned | 770.10 | 1,083.74 | 0.8783 |
| GBM Post-PIF Tuned | 757.09 | 1,051.98 | 0.8790 |


OLS stands for Ordinary Least Squares, RidgeCV stands for Cross Validated Ridge Regression, RF stands for Random Forest, GBM stands for Gradient Boosting Machine.

MAE is the Mean Absolute Error, RMSE is the Root Mean Square Error, R² is the coefficient of determination.

For all of the models there are no signs of overfitting or test R-squared being significantly higher than the train R-squared score, hence all models show good 'behavior'. In terms of goodness of fit, all models perform relatively similarly. There is an increase of about 2% in R-squared when moving from linear to nonlinear models. However, there are substantial differences when comparing MAE and RMSE across models. Since the task is price prediction, we rely primarily on MAE. Using our best-performing model (GBM), predictions are on average off by only €757.09. GBM also performs best when evaluated using RMSE, indicating its ability to produce fewer severe pricing errors. Note that RMSE is larger than MAE for all models, indicating that when poor price predictions occur, the errors can sometimes be large.

Overall, the results show clearly that GBM is the best model to predict car prices with the given dataset.

## How to Run 
1. Make sure you have Python3 installed in your personal computer.
2. Clone the repository.
3. Open the repository using your preferred IDE(e.g. VS Code).
4. Navigate to the project folder where you saved the repo:
   cd Predictions-of-Car-Prices-Using-Linear-and-Machine-Learning-Models
5. Create a virtual environment: python -m venv your_preferred_venv_name
6. Activate the virtual environment according to the operating system you use(e.g. Windows).
7. Install the necessary libraries in the activated virtual environment:
   python -m pip install -r requirements.txt
8. Run the python scripts in this order: toyota_cleaned.py -> prices_insights.py -> feature_engineering.py -> linear_modeling.py -> nonlinear_modeling.py

\* Please note that nonlinear_modeling.py includes grid searches that take a significant time to run. Before running the file, open it and find the sections that include GridSearchCV (use can use CTRL+F and write GridSearchCV to find the sections). Then, read the comments above these sections to modify the parameter grids in order to run the code faster.
