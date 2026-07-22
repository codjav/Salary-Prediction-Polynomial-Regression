# Salary Prediction using Polynomial Regression

## Objective

Develop a Polynomial Regression model to predict employee salaries based on position level.

---

## Dataset

https://www.kaggle.com/datasets/akram24/position-salaries

---

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## Methodology

1. Load dataset
2. Explore data
3. Check missing values
4. Select feature and target
5. Split data into training and testing sets
6. Transform feature using Polynomial Features (Degree = 3)
7. Train Polynomial Regression model
8. Predict salaries
9. Evaluate using MAE, MSE and R² Score
10. Plot Polynomial Regression Curve

---

## Results

- Mean Absolute Error (MAE): **70635.25**
- Mean Squared Error (MSE): **6263853282.86**
- R² Score: **0.8763**

---

## Observations

- Polynomial Regression models the nonlinear salary trend effectively.
- The fitted curve follows the original data closely.
- The model explains approximately **87.63%** of the variance in salary.

---

## Conclusion

Polynomial Regression provides a better fit for this dataset because the relationship between position level and salary is nonlinear. Unlike Linear Regression, which assumes a straight-line relationship, Polynomial Regression captures curved trends by introducing polynomial features. The model achieved a strong R² score of **0.8763**, indicating good predictive performance. Position level has a significant impact on salary, with higher levels showing rapidly increasing salaries. A major advantage of Polynomial Regression is its ability to model complex nonlinear relationships while remaining simple to implement. However, selecting an excessively high polynomial degree may lead to overfitting.
