import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Given data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([52, 55, 61, 70, 82])

linear_model = LinearRegression()
linear_model.fit(x, y)

b0 = linear_model.intercept_
b1 = linear_model.coef_[0]

print("Model A: Simple Linear Regression")
print(f"Equation: y = {b0:.2f} + {b1:.2f}x")


x_test = np.array([[6]])
y_pred_linear = linear_model.predict(x_test)
print(f"Prediction at x = 6: {y_pred_linear[0]:.2f}")


y_train_pred_linear = linear_model.predict(x)
mse_linear = mean_squared_error(y, y_train_pred_linear)
print(f"Training MSE (Model A): {mse_linear:.2f}\n")


poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)

poly_model = LinearRegression()
poly_model.fit(x_poly, y)

print("Model B: Polynomial Regression (Degree 4)")
print("Coefficients:")
print(f"w0 = {poly_model.intercept_:.2f}")
for i, coef in enumerate(poly_model.coef_[1:], start=1):
    print(f"w{i} = {coef:.2f}")


x_test_poly = poly.transform(x_test)
y_pred_poly = poly_model.predict(x_test_poly)
print(f"\nPrediction at x = 6: {y_pred_poly[0]:.2f}")


y_train_pred_poly = poly_model.predict(x_poly)
mse_poly = mean_squared_error(y, y_train_pred_poly)
print(f"Training MSE (Model B): {mse_poly:.6f}")