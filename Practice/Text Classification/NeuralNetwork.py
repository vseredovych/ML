import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_surface(cls, x_1, x_2, ax=None, threshold=0.5, contourf=False):
    xx1, xx2 = np.meshgrid(np.linspace(x_1.min(), x_1.max(), 100), 
                           np.linspace(x_2.min(), x_2.max(), 100))

    X_pred = np.c_[xx1.ravel(), xx2.ravel()]
    pred = cls.predict_proba(X_pred)[:, 0]
    Z = pred.reshape((100, 100))
    if ax is None:
        ax = plt.gca()
    ax.contour(xx1, xx2, Z, levels=[threshold], colors='black')
    ax.set_xlim((x_1.min(), x_1.max()))
    ax.set_ylim((x_2.min(), x_2.max()))
    
def plot_data(A, b, test = False):
    positive_indices = np.where(b == 1)[0]
    negative_indices = np.where(b == 0)[0]
    
    plt.scatter(A[positive_indices, 0], A[positive_indices, 1], marker='x', c= 'yellow' if test else 'green')
    plt.scatter(A[negative_indices, 0], A[negative_indices, 1], marker='+', c= 'blue' if test else 'red')

class NeuralNet:
    """
    NN for binary classification
    Attributes:
    ...
    """
    # hidden_layer_sizes=(100, 50,)
    def __init__(self, hidden_layer_sizes, normalize = True, learning_rate = 0.01, num_iter = 30000, batch_size=64, optimization='', momentum=0.9):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.normalize = normalize
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimization = optimization
        self.momentum = momentum
        self.batch_size = batch_size
        
    def __normalize(self, X, mean = None, std = None):
        """
        Зверніть увагу, що нормалізація вхідних даних є дуже важливою для швидкодії нейронних мереж.
        """
        n = X.shape[0]
    
        if mean is None:   
            mean = np.zeros([n, 1])
        if std is None:
            std  = np.ones([n, 1])
        
        for i in range(n):
            if (np.std(X[:, i]) != 0):
                if mean is None:
                    mean[i] = np.mean(X[:, i])
                if std is None:
                    std[i] = np.std(X[:, i])
        
        X_new = (X - mean) / std
        return X_new, mean, std
    

    def __sigmoid(self, Z):
        """
        В наступних практичних потрібно буде додати підтримку й інших активаційних функцій - це один з гіперпараметрів. 
        Їх можна вибирати для всіх шарів одночасно або мати різні активаційні функції на кожному з них.
        """
        return 1 / (1 + np.exp(-Z))
    
    def __sigmoid_derivative(self, Z):
        z = self.__sigmoid(Z)
        return np.multiply(z, (1 - z))
    
    def __softmax(self, Z):
        exp_z = np.exp(Z)
        return exp_z / exp_z.sum(axis=0, keepdims=True)
    
    def __cross_entropy(self, A, Y):
        return - np.sum(Y * np.log(A), axis=1)
    
    def __initialize_parameters(self, n_x, n_y):
        self.parameters = {}
        n = len(n_x)
        
        for i in range(1, len(n_x)):
            W = np.random.randn(n_x[i], n_x[i - 1]) * 0.01
            b = np.zeros((n_x[i], 1))
            self.parameters.update({f"W{i}": W, f"b{i}": b})

        W = np.random.randn(n_y, n_x[n-1]) * 0.01
        b = np.zeros((n_y, 1))
        
        self.parameters.update({f"W{n}":W, f"b{n}":b})
                    
        for i in range(1, len(n_x) + 1):
            self.parameters.update({f"VdW{i}": 0, f"Vdb{i}": 0})
            
    def __forward_propagation(self, X):
        num_layers = len(self.hidden_layer_sizes)
        cache = self.parameters.copy()
        
        A = X
        for i in range(1, num_layers + 2):
            if i == num_layers + 1:
                W = self.parameters[f"W{i}"]
                b = self.parameters[f"b{i}"]
                Z = np.dot(W, A) + b
                A = self.__softmax(Z)
            else:            
                W = self.parameters[f"W{i}"]
                b = self.parameters[f"b{i}"]
                Z = np.dot(W, A) + b
                A = self.__sigmoid(Z)
            cache.update({f"Z{i}": Z})
            cache.update({f"A{i}": A})
        return A, cache

    def compute_cost(self, A, Y):
        J = -np.mean(Y.T * np.log(A.T))
        return J
    
    def __backward_propagation(self, X, Y, cache):
        m = X.shape[1]
        n = X.shape[0]
        num_layers = len(self.hidden_layer_sizes)
        
        grads = {}
        
        for i in range(num_layers+1, 0, -1):
            W = cache[f"W{i}"]
            b = cache[f"b{i}"]
            Z = cache[f"Z{i}"]
            if i == num_layers + 1:
                A = cache[f"A{i}"]
                A_next = cache[f"A{i - 1}"]
                
                dZ = A - Y
                dW = 1. / m * np.dot(dZ, A_next.T)
                db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
            elif (i == 1):
                W_prev = cache[f"W{i + 1}"]
                A = cache[f"A{i}"]
                
                dA = np.dot(W_prev.T, dZ)
                dZ = np.multiply(dA, self.__sigmoid_derivative(A))
                dW = 1. / m * np.dot(dZ, X.T)
                db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
            else:
                W_prev = cache[f"W{i + 1}"]
                A = cache[f"A{i}"]
                A_next = cache[f"A{i - 1}"]

                dA = np.dot(W_prev.T, dZ)
                dZ = np.multiply(dA, self.__sigmoid_derivative(A))
                dW = 1. / m * np.dot(dZ, A_next.T)
                db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
            grads.update({f"dZ{i}":dZ, f"dW{i}":dW, f"db{i}":db})
        return grads
    
    def __update_parameters(self, grads, iteration):
        num_layers = len(self.hidden_layer_sizes)
        
        for i in range(1, num_layers + 2):
            W = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]
            
            dW = grads[f"dW{i}"]
            db = grads[f"db{i}"]
            if self.optimization == 'momentum':
#                 print(1 - self.momentum ** iteration)
                self.parameters[f"VdW{i}"] = (self.momentum * self.parameters[f"VdW{i}"] + (1 - self.momentum) * dW / (1 - self.momentum ** (iteration + 1)))
                self.parameters[f"Vdb{i}"] = (self.momentum * self.parameters[f"Vdb{i}"] + (1 - self.momentum) * db / (1 - self.momentum ** (iteration + 1)))

                self.parameters[f"W{i}"] = W - self.learning_rate * self.parameters[f"VdW{i}"]
                self.parameters[f"b{i}"] = b - self.learning_rate * self.parameters[f"Vdb{i}"]

            else:
                self.parameters[f"W{i}"] = W - self.learning_rate * dW
                self.parameters[f"b{i}"] = b - self.learning_rate * db
    
    def __create_mini_batches(self, X, Y, batch_size):
        m = X.shape[1]
        x_columns = X.shape[0]
        
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]        
        shuffled_Y = Y[:, permutation]
        data = np.vstack((shuffled_X, shuffled_Y))

        mini_batches = []
        n_minibatches = m // batch_size
        i = 0

        for i in range(n_minibatches + 1): 
            mini_batch = data[:, i * batch_size:(i + 1)*batch_size] 
            X_mini = mini_batch[:x_columns, :] 
            Y_mini = mini_batch[x_columns:, :]
            mini_batches.append((X_mini, Y_mini))
            
            if data.shape[0] % batch_size != 0:
                mini_batch = data[:, i * batch_size:m] 
                X_mini = mini_batch[:x_columns, :]
                Y_mini = mini_batch[x_columns:, :]
                mini_batches.append((X_mini, Y_mini)) 
        return mini_batches
    
    def fit(self, X_vert, Y_vert, epsilon=1e-08, print_cost = True):
        
        X, Y = X_vert.T, np.expand_dims(Y_vert, axis=1).T
        
        if self.normalize:
            X, self.__mean, self.__std = self.__normalize(X)
        
        costs = []
        
        m = X.shape[1]
        n_x = (X.shape[0],) + self.hidden_layer_sizes
        n_y = Y.shape[0]
        
        self.__initialize_parameters(n_x, n_y)
        
        for i in range(self.num_iter):
            mini_batches = self.__create_mini_batches(X, Y, self.batch_size) 
            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch 
                A, cache = self.__forward_propagation(X_mini)

                cost = self.compute_cost(A, Y_mini)

                grads = self.__backward_propagation(X_mini, Y_mini, cache)

                self.__update_parameters(grads, i)

                costs.append(cost)

                if print_cost and i % 1000 == 0:
                    print("{}-th iteration: {}".format(i, cost))
                    if i > 1:
                        print(f"Delta: {costs[-2] - costs[-1]}")

                if i > 1 and abs(costs[-2] - costs[-1]) < epsilon:
                    break
                
        if print_cost:
            plt.plot(costs)
            plt.ylabel("Cost")
            plt.xlabel("Iteration, *1000")
            plt.show()

    def predict_proba(self, X_vert):
        X = X_vert.T
        if self.normalize:
            X, _, _ = self.__normalize(X, self.__mean, self.__std)
        
        probs = self.__forward_propagation(X)[0]
        return probs.T
    
    def predict(self, X_vert):
        positive_probs = self.predict_proba(X_vert)
        y_pred = self.likehood_func(positive_probs)
        return y_pred  

    def likehood_func(self, z):
        return z.argmax(axis=1)