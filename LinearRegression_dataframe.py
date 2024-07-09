import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler



class LinearRegression:
    def __init__(self, data, target,  learning_rate=0.00001) -> None:
        self.learning_rate = learning_rate
        self.data = np.array(data)
        self.label = np.array(target).reshape(-1, 1)
        self.samples, num_features = self.data.shape
        self.weight = np.random.rand(num_features, 1)
        self.bias = np.random.rand()
        
    def prediction(self):
        return np.dot(self.data, self.weight) + self.bias
    
    def MSE(self):
        errors = self.prediction() - self.label
        return np.mean(np.square(errors))
    
    def train(self, iterations, batch):
        for i in range(iterations):
            y_pred = self.prediction()
            dw = (-2 / self.samples) * np.dot(self.data.T, (y_pred - self.label))
            db = (-2 / self.samples) * np.sum(y_pred - self.label)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % batch == 0:
            # if i == batch:
                print(f"Iteration {i}: Weight={self.weight.squeeze()}, Bias={self.bias}, MSE={self.MSE()}")
    
