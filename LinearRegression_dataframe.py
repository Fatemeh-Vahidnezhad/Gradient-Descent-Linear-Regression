import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, data, learning_rate=0.00001) -> None:
        label = np.array([row[-1] for row in data])
        self.data = np.array([row[:-1] for row in data])
        self.samples, self.num_features = self.data.shape
        self.label = label.reshape(self.samples, 1)

        self.weight = np.random.rand(self.num_features, 1)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def prediction(self):
        y_pred = np.dot(self.data, self.weight) + self.bias
        return y_pred
    
    def MSE(self):
        errors = self.prediction() - self.label
        return np.mean(np.square(errors))
    
    def gradient_descent(self, iterations=20000):
        for i in range(iterations):
            y_pred = self.prediction()
            dw = (-2 / self.samples) * np.dot(self.data.T, (y_pred - self.label))
            db = (-2 / self.samples) * np.sum(y_pred - self.label)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % 1000 == 0:
                print(f"Iteration {i}: Weight={self.weight.squeeze()}, Bias={self.bias}, MSE={self.MSE()}")

data = {
    'Height': [172, 178, 165, 190, 155],
    'Weight': [65, 70, 60, 90, 55],
    'Age': [25, 41, 30, 35, 22],
    'Score': [88, 92, 77, 85, 91],
    'Health_Score': [75, 82, 69, 88, 72]  # This is the label for prediction.
}
df = pd.DataFrame(data)
scaler = StandardScaler()
df = scaler.fit_transform(df)

lr = LinearRegression(df)
lr.gradient_descent()
