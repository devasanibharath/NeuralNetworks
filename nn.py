from __future__ import division
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from warnings import warn
from Canvas import Line
import csv
class Neural_Network():
    
    def __init__(self, layers, random_seed=None, learning_rate=1, opt=False, \
                 epsilon=.15, activation_func='sigmoid', epochs=200000, \
                 check_gradients=False, C=1, batch_size=100):
 
    
        self.weights = []
        self.cost = [0]
        self.tdeltas = []
        self.deltas = range(len(layers) - 1)
        self.a = range(len(layers))
        self.epsilon = epsilon
        self.layers = layers
        self.learning_rate = learning_rate
        self.__stoping_threshold = 1e-13
        self.opt = opt
        self.epochs = epochs
        self.check_gradients = check_gradients
        self.C = C
        self.batch_size = batch_size
        self.best_weights = None
        self.lowest_cost = None
        
        if activation_func == 'sigmoid':
            self.activation_func = self.sigmoid
            self.activation_gradient = self.sigmoid_gradient
        elif activation_func == 'tanh':
            self.activation_func = self.tanh
            self.activation_gradient = self.tanh_gradient

        if random_seed is not None:
            np.random.seed(random_seed)
           
        for i, layer_size in enumerate(layers[:-1]):
            self.weights.append(self.get_random_weights(layers[i:i+2]))
            self.tdeltas.append(np.zeros((layer_size + 1, layers[i+1])))
                                
    
    def get_random_weights(self, x):
        return np.random.rand(x[0] + 1, x[1]) * 2 * self.epsilon - self.epsilon
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def sigmoid_gradient(self, x):
        return np.multiply(x, 1-x)
    
    def tanh_gradient(self, x):
        return 1.0 - x**2
    
    def feed_forward(self):
        for i, theta in enumerate(self.weights):
            z = np.dot(self.a[i], theta)
            self.a[i+1] = np.c_[np.ones(len(z)), self.activation_func(z)]
        # Remove column of ones in last layer  
        self.a[-1] = self.a[-1][:,1:]
    
    
    def backpropagation(self, change_weights=True):
        self.deltas[-1] = np.multiply(self.a[-1] - self.__y_encoded_mat[self.__random_mini_batch], self.activation_gradient(self.a[-1]))
        
        for i in range(-1, -len(self.deltas), -1):
            self.deltas[i-1] = np.multiply(np.dot(self.deltas[i], self.weights[i].T), self.activation_gradient(self.a[i-1]))[:, 1:]
            
        for i, D in enumerate(self.tdeltas):
            regularized_weights = self.weights[i] * self.C
            regularized_weights[:,0] = 0
            self.tdeltas[i] = (np.dot(self.a[i].T, self.deltas[i])  + regularized_weights) / len(self.__random_mini_batch) # len(self.__y_encoded_mat)
            if change_weights:
                self.weights[i] -= self.learning_rate * self.tdeltas[i] 
    
    
    def convert_weights_back_to_matrix(self, x):

        cum_sum = [0]
        for i, layer in enumerate(self.layers[:-2]):

            cum_sum.append((layer + 1) * self.layers[i + 1] + cum_sum[-1])
            self.weights[i] = x[cum_sum[-2]:cum_sum[-1]].reshape(layer+1, self.layers[i + 1])
        
        self.weights[-1] = x[cum_sum[-1]:].reshape(self.layers[i + 1] + 1, self.layers[i + 2])
    
    
    def __cost_func_opt(self, x, *args):
        X,y = args
        self.convert_weights_back_to_matrix(x)
        
        self.feed_forward()
        return self.__cost_function_se()
    
    
    def __gradf(self, x, *args):
        X,y = args
        self.convert_weights_back_to_matrix(x)
        self.set_mini_batch(X)
        self.feed_forward()
        self.backpropagation(change_weights=False)
        
        tdeltas = self.tdeltas[0].ravel()
    
        for D in self.tdeltas[1:]:
            tdeltas = np.r_[tdeltas, D.ravel()]
        return tdeltas
    

    def __grad_check(self, theta, X):
        grads = np.zeros(len(theta))
        epsilon = 1e-4
        for i, grad in enumerate(grads):            
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            self.convert_weights_back_to_matrix(theta_plus)
            self.set_mini_batch(X)
            self.feed_forward()
            cost_plus = self.__cost_function_se()
            
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            self.convert_weights_back_to_matrix(theta_minus)
            self.feed_forward()
            cost_minus = self.__cost_function_se()
            
            grads[i] = (cost_plus - cost_minus) / (2*epsilon)
        
        self.convert_weights_back_to_matrix(theta)
        self.feed_forward()
        self.backpropagation()
        tdeltas = self.tdeltas[0].ravel()
        for D in self.tdeltas[1:]:
            tdeltas = np.r_[tdeltas, D.ravel()]
        
        print "manual grad check MSE", np.sum((grads - tdeltas)**2) / len(grads) 
       
    
    def __cost_function_mle(self, y, h, m):
        return -1./m * np.sum(y * np.log(h) + (1-y) * np.log(1-h))

    def __cost_function_se(self):
        regularized_sum = 0
        for weight in self.weights:
            regularized_sum += np.sum(weight[:,1:]**2)
#         regularized_sum = regularized_sum * self.C / (2*len(self.__y_encoded_mat))
        regularized_sum = regularized_sum * self.C / (2*self.batch_size)

        
#         return 1/len(self.__y_encoded_mat) * np.sum((self.__y_encoded_mat - self.a[-1]) ** 2) / 2 + regularized_sum
        return 1/self.batch_size * np.sum((self.__y_encoded_mat[self.__random_mini_batch] - self.a[-1]) ** 2) / 2 + regularized_sum

    def predict(self, X):
        a = np.c_[np.ones(len(X)), X]
        for i, theta in enumerate(self.weights):
            z = np.dot(a, theta)
            a = np.c_[np.ones(len(z)), self.activation_func(z)]
        
        if len(self.classes_) > 2:
            soft_max = np.argmax(a[:,1:], axis=1)
            reverse_lookup = {v:k for k,v in self.__encoded_classes.iteritems()}
            returned_prediction = np.array([reverse_lookup[x] for x in soft_max])
        else:
            returned_prediction = np.round(a[:,1:])
            
        return returned_prediction
    
    def y_encode_values(self, y):
        if np.ndim(y) == 2 and y.shape[1] != 1:
            raise ValueError("bad y shape {0}. Make y 1 dimension or have 1 as its second dimension".format(y.shape))
        
        self.classes_ = np.unique(y)
        self.__encoded_classes = {cls:i for i, cls in enumerate(self.classes_)}
        
        if type(y) == np.ndarray:
            y = y.ravel()
        self.__y_encoded = np.atleast_2d(np.array([[self.__encoded_classes[cls] for cls in y]]).T)
        
        self.__y_encoded_mat = self.__y_encoded.copy()
        if len(self.classes_) > 2:
            self.__y_encoded_mat = np.zeros((len(self.__y_encoded), len(self.classes_)))
            for i, y_encoded in enumerate(self.__y_encoded):
                self.__y_encoded_mat[i, y_encoded] = 1
                
                
    def set_mini_batch(self, X):
        if self.batch_size > len(X):
            warn("Batch size is greater than number of training examples. " +
                 "Will use a batch size equal to the number of training examples.")
            self.batch_size = len(X)
        
        self.__random_mini_batch = np.random.choice(range(len(X)), self.batch_size, False)        
        self.a[0] = np.c_[np.ones(self.batch_size), X[self.__random_mini_batch]]
        
    def get_cost(self, X, y):
        current_cost = self.__cost_function_se()
        self.cost.append(current_cost)
        if current_cost < self.lowest_cost or self.lowest_cost is None:
            self.lowest_cost = current_cost
            self.best_weights = [weight.copy() for weight in self.weights]
            self.best_accuracy = self.score(X, y)
        
        
    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError('X and y have incompatible shapes. X has ' + str(len(X)) + ' examples but ' +
                             'y has ' + str(len(y)) + '.')
        
        
            
        self.y_encode_values(y)
        self.a[0] = np.c_[np.ones(len(X)), X]
        
        
        if self.opt:
            unraveled_thetas = self.weights[0].ravel()
            for theta in self.weights[1:]:
                unraveled_thetas = np.r_[unraveled_thetas, theta.ravel()]
                
            if self.check_gradients:
                
                print "gradient check with scipy", optimize.check_grad(self.__cost_func_opt, self.__gradf, unraveled_thetas, \
                                                            X, y)
                self.__grad_check(unraveled_thetas, X)
            
            theta_opt,min_val,c,d, e = optimize.fmin_cg(self.__cost_func_opt, fprime=self.__gradf, x0 = unraveled_thetas,\
                                                        args = (X, y), full_output=1, gtol=1e-8)
            
            self.convert_weights_back_to_matrix(theta_opt)
            

        else:
            for i in range(self.epochs):
                
                # Get random batch
                self.set_mini_batch(X)

                self.feed_forward()

                self.get_cost(X, y)

                self.backpropagation()
            self.end_weights = self.weights[:]
            self.weights = self.best_weights
            print "lowest cost was ", self.lowest_cost
                

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def final_weights(self):
        return self.best_weights 

if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'   
    X = []
    y =[]
    train_file = open(train_file,'r')
    for elements in train_file:
        element = elements.split()
        X_elem = [float(element[0]),float(element[1])]
        y_elem = [int(element[2])]
        X.append(X_elem)
        y.append(y_elem)
    
    X = np.array(X)
    y = np.array(y)
    nn = Neural_Network([2,10,5,1], learning_rate=1, C=0, opt=False, check_gradients=True, batch_size=200, epochs=10000)
    nn.fit(X,y)    
#     xx, yy = np.meshgrid(np.linspace(-5,5,100),np.linspace(-5,5,100))
#     #xx, yy = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
#     Z = np.round(nn.predict(np.c_[xx.ravel(), yy.ravel()]))
#     Z = Z.reshape(xx.shape)
    weights = nn.final_weights()
    X_out = []
    y_out = []
    test_file = open(test_file,'r')
    for elements in test_file:
        element = elements.split()
        X_out_elem = [float(element[0]),float(element[1])]
        y_out_elem = [int(element[2])]
        X_out.append(X_out_elem)
        y_out.append(y_out_elem)
    testlabels_out = nn.predict(X_out)
    print testlabels_out
    
    print "Neural Net Accuracy is " + str(np.round(nn.score(X_out,y_out),2))
    del X,y,X_out,y_out