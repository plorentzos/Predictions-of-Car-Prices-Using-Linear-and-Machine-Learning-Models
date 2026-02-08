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
The following relate to toyota_cleaned.py script.
* The raw dataset is loaded and the first 5 rows are printed to get a sense of what the data consists of.
* Data Inspection and column renaming
* Then I rename column names for better interpretability.
* I transform the data according to what is more optimal to perform the analysis.
* I check for missing values and duplicates. I overview the data type of each variable and the dimensions of the whole dataset.
* I do some final checks to see if the dataset is nice and tidy before I start the analysis.
* Finally in each file, besides the correlations one, I create a risk measure and visualize it accordingly. The correlations_between_measures.py contains the correlations calculation and visualization between Altman's Z score, Merton's DD ,and Value at Risk risk measures.

Note that additional comments regarding the code and the analysis can be found in the .py files.

## Modeling Assumptions
* Daily and yearly returns are being calculated using the standard finance textbook formulas with continuous compounding.
* For Altman's Z Score refer to the non-manufacturer bankruptcy model on https://en.wikipedia.org/wiki/Altman_Z-score
* For Merton's Distance to Default refer to "Forecasting Default with Merton Distance to Default Model" (Bharath et. al 2008) , particularly section 2.3.
* Value At Risk and Expected Shortfall are calculated using the standard textbook formulas. In any case, you can refer to Quantitative Risk Management book (McNeil et. al) chapter 2.3.

## Insights
* According to the cross-sectional  average Z-Score metric, our portfolio is considered to be in the 'safe zone' indicating a negligible risk of default. However, it seems that there is an downward trend over time, showing that these companies have gotten less financially "healthy" over time. Of course, as seen in the Z-score yearly table, some of the companies in our portfolio are considered to be in the 'distress zone' as they have Z-scores less than 1.10. For these companies, we can cross-check their risk of default using Merton's DD measure (or the naive default probabilities derived using it). If both measures indicate that the company is in the 'distress zone' we should consider excluding it from our portfolio of stocks if we are a risk-averse investor.
  
* According to the cross-sectional average Merton's naive Distance to Default (DD) measure, our portfolio of stocks seems to have higher default risk over time with the lowest point to be in 2020 due to the COVID-19 pandemic. Calculated naive DD probabilities are also useful to get a sense of the probability of default since Merton's naive DD is measured in standard deviations.

* Assume we hold 1 billion euros worth of each stock, i.e., we have invested a total of 50 billion euros in our portfolio of stocks. According to the cross-sectional average Value at Risk (VaR) and Expected Shortfall (ES), our portfolio seems to have increased market risk to extreme losses over time with a spike in 2022 due to the COVID-19 pandemic.
  
* Looking at the correlation between the created risk measures we can see that Altman's Z score and VaR have very weak correlation. This is expected as VaR relies only one daily stock prices whereas Altman's Z score relies on yearly fundamental company values.  We also see that during periods of distress (CoVid-19) correlations between Z score and VaR are virtually zero. DD and VaR are negatively correlated since a decreasing DD means the firm gets closer to default which signifies increased credit risk, which in turn leads to higher potential losses (VaR). We also see that during periods of distress(Covid-19) correlations between DD and VaR get less negative.Z score and DD have positive correlation as they are both default risk measures, but the level of correlation is weak since they are built using different variables and assumptions. 

## How to Run 
1. Make sure you have at least Python version 3.9+ installed in your personal computer.
2. Clone the repository.
3. Open the repository using your preferred IDE(e.g. VS Code).
4. Navigate to the project folder where you saved the repo:
   cd Default-and-Market-Risk-Assessment-of-Public-US-Companies
5. Create a virtual environment using uv: python -m uv venv
6. Activate the virtual environment according to the operating system you use(e.g. Windows, Linux, macOS).
7. Install the necessary libraries in the activated virtual environment:
   python -m pip install -r requirements.txt
8. Run the python scripts in this order: toyota_cleaned.py -> prices_insights.py -> feature_engineering.py -> linear_modeling.py -> nonlinear_modeling.py
