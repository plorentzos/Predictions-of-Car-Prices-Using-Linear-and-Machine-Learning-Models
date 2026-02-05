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
In this project I use daily prices and financial statement data of 50 public U.S companies  to create measures in order to assess their exposure to default and market risk over a time horizon of  9 years.

The analysis includes data cleaning, exploratory data analysis and the creation, assessment and visualization of robust risk-related financial metrics. 

The analysis is performed in Python. 

## Data
There are 3 datasets used for the analysis:

* The first dataset includes the daily prices of the companies. 
* The second dataset includes annual financial statements variables and accounting ratios. 
* The third dataset includes yearly equity and debt values of companies.

The datasets include information about 50 public U.S companies, continuous members of the S&amp;P 500 Index, for the period spanning from 2012 to 2022.
The companies included represent the following sectors: Information Technology, Communication Services, Healthcare, Consumer Staples, Consumer Discretionary, Industrials, Energy and Utilities.
Including different sectors allows us to obtain a more diversified portfolio. Also, it opens the door for a sector analysis if deemed necessary.

Note that datasets that are generated during the analysis can be found in data_created folder in the data folder.
Note that graphs generated during the analysis can be found in the graphs folder.
Note that you can always delete the data_created and graphs folder. They will get generated again when you run the .py files.

## Tools and Technologies
* Python (Libraries: Pandas, NumPy, SciPy, Matplotlib)

## Analysis Workflow
* First, I inspect the raw data.
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
8. Run the python scripts in this order: altman_z_score.py -> merton_dd.py -> var_and_es.py -> correlations_between_measures.py
