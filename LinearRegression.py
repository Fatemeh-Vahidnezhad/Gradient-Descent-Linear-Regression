import numpy as np

class LinearRegression:
    def __init__(self, data, label, learning_rate=0.0001) -> None:
        self.data = data
        self.label = label
        self.samples, self.num_features = self.data.shape
        self.weight = np.random.rand(self.num_features, 1)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def prediction(self):
        return np.dot(self.data, self.weight) + self.bias
    
    def MSE(self):
        errors = self.prediction() - self.label
        return np.mean(np.square(errors))
    
    def gradient_descent(self, iterations=100000):
        for i in range(iterations):
            y_pred = self.prediction()
            dw = (-2 / self.samples) * np.dot(self.data.T, (y_pred - self.label))
            db = (-2 / self.samples) * np.sum(y_pred - self.label)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % 100 == 0:
                print(f"Iteration {i}: Weight={self.weight.squeeze()}, Bias={self.bias}, MSE={self.MSE()}")

# Example usage
data = np.array([[7], [10], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]])
label = np.array([[8], [11], [14],[15], [16], [17], [18], [19], [20], [21], [22], [23], [24]])

lr = LinearRegression(data, label, learning_rate=0.00001)
lr.gradient_descent(iterations=1000)
