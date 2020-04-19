# Temperature-Prediction-of-Cities-of-India

This is a Machine Learining approach using the following Regression models-
1) Random Forest Regression
2) Support Vector Regression
3) Multi Linear Regression
4) Polynomial Regression

To predict the temperature of different cities-
1) Chennai
2) Delhi
3) Mumbai
4) Kolkata

on any given date(mm-dd-yyyy)

The temperature dataset of these cities were obtained from https://www.kaggle.com/riturajsaha/temperature-of-different-cities-of-india

R Squared value of different models are used to compare them. R-squared is a statistical measure that represents the goodness of fit of a regression model. The ideal value for r-square is 1. The closer the value of r-square to 1, the better is the model fitted.

Below is cmaprison of different regression models o diffierent cities in the dataset on the basis of R-Squared value.

| First Header  | Chennai | Delhi | Mumbai | Kolkata |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Random Forest Regression  | 0.1120  | 0.4816  | 0.0223  | 0.3661  |
| Support Vector Regression  | 0.1390  | 0.3564  | 0.7058  | 0.2380  |
| Multi-linear Regression | 0.0045  | 0.0017  | -0.0004  | 0.0006  |
| Polynomial Regression  | 0.0045  | 0.0017  | -0.0004  | 0.0006  |

The temeperature prediction of a city is obtained with an accuracy of 0.7058 R Squared value from SVR Model in Mumbai dataset.
