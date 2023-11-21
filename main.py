import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Random Forest Regression using scikit-learn
regressor_rf = RandomForestRegressor(random_state=0, n_estimators=10)
regressor_rf.fit(X, y)

# Convert X to PyTorch tensor for plotting
X_tensor_plot = torch.tensor(np.arange(min(X), max(X), 0.1).reshape(-1, 1), dtype=torch.float32)

# Predicting the salary for the plot
y_pred_rf_plot = regressor_rf.predict(X_tensor_plot)

# Convert predictions back to numpy arrays
X_plot_np = X_tensor_plot.numpy()

# Visualizing the Random Forest Regression Results with Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.flatten(), y=y.flatten(), color='red', label='Actual Data')
sns.lineplot(x=X_plot_np.flatten(), y=y_pred_rf_plot.flatten(), color='blue', label='Random Forest Prediction')
plt.title('Random Forest Regression with Seaborn')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualizing the Random Forest Regression Results with Plotly
fig = px.scatter(x=X.flatten(), y=y.flatten(), title='Random Forest Regression with Plotly',
                 labels={'x': 'Position Level', 'y': 'Salary'})
fig.add_scatter(x=X_plot_np.flatten(), y=y_pred_rf_plot.flatten(), mode='lines', name='Random Forest Prediction',
                line=dict(color='blue'))
fig.show()

# Deep Learning Regression using Keras
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)

# Creating a deep learning model with Keras
model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# Predicting with the trained model
y_pred_keras = model.predict(X_scaled)

# Inverse transform for visualization
y_pred_keras = scaler.inverse_transform(y_pred_keras)

# Visualizing the Deep Learning Regression Results with Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.flatten(), y=y.flatten(), color='red', label='Actual Data')
sns.lineplot(x=X.flatten(), y=y_pred_keras.flatten(), color='green', label='Keras Prediction')
plt.title('Deep Learning Regression with Seaborn')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.legend()
plt.show()
