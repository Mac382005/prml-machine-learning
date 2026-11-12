import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Reshape X into a column vector
X = X.reshape(-1, 1)

# Define the degree of the polynomial, you can try other values
degree = 3

# Create polynomial features using PolynomialFeatures from the scikit-learn
# Transform X to polynomial features using fit_transform
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Create and train a linear regression model
# Fit it to the polynomial features X_poly and the target variable y
model = LinearRegression()
model.fit(X_poly, y)

# Generate data for plotting
# Generate a range of X values using linspace, transform it to polynomial features
# Predict the corresponding y values using model.predict
X_plot = np.linspace(0, 6, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model.predict(X_plot_poly)

# Plot the original data and the fitted polynomial curve using matplotlib
plt.scatter(X, y, label='Original Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Fit of Degree {}'.format(degree))
plt.legend()
plt.show()




# # Import package
# # NumPy for numerical operations
# # LinearRegression from scikit-learn for building a linear regression model.
# import numpy as np
# from sklearn.linear_model import LinearRegression
#
# # Dataset, define input X and the corresponding target variable y using NumPy arrays
# # Here X is data we collected (some features of houses), y is the house price
# X = np.array([[1200, 2, 1, 1995],
# [1500, 3, 2, 2002],
# [1800, 3, 2, 1985],
# [1350, 2, 1, 1998],
# [2000, 4, 3, 2010]])
# y = np.array([250, 320, 280, 300, 450])
#
# # Create and train the model
# # Create an instance of the LinearRegression model and fit it to the data (X,y)
# model = LinearRegression()
# model.fit(X, y)
#
# # Prediction. After training, we can new use it to predict the prices for new houses
# new_data = np.array([[1650, 3, 2, 2005],
# [1400, 2, 1, 2000]])
# predicted_prices = model.predict(new_data)
# print("Predicted prices:", predicted_prices)