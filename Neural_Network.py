"""Classic Neural Network"""

#Importing libraries
import numpy as np

def sigmoid(x):
    #Sigmoid activation function: f(x) = 1/(1 + e^(-x))
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    #Derivative of sigmoid fuinction: f'(x) = f(x)*(1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_error(y_true, y_pred):
    #y_true and y_pred are numpy arrays of same length
    return ((y_true - y_pred)**2).mean()

class NeuralNetwork:
    #Neural network with two neurons
    def __init__(self):
        #weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        #Bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def feedforward(self, x):
        #x is a numpy array of two elements
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_true):
        #Setting the learning rate
        learn_rate = 0.1
        #setting the number of epochs
        epochs = 1000
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_true):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                #Calculate the partial derivative
                #Partial derivative with respective to w1
                
                dl_dy_pred = -2 * (y_true - y_pred)
                
                #Neuron o1
                dy_pred_dw5 = h1 * derivative_sigmoid(sum_01)
                dy_pred_dw6 = h2 * derivative_sigmoid(sum_01)
                dy_pred_b3 = derivative_sigmoid(sum_o1)
                
                dy_pred_dh1 = self.w5 * derivative_sigmoid(sum_o1)
                dy_pred_dh2 = self.w6 * derivative_sigmoid(sum_o1)
                
                #Neuron h1
                d_h1_dw1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_dw2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_db1 = derivative_sigmoid(sum_h1)
                
                #Neuron h2
                d_h2_dw3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_dw4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_db2 = derivative_sigmoid(sum_h2)
                
                #Calculate and update the weights and biases
                #Neuron 1
                self.w1 -= learn_rate * dl_dy_pred * dy_pred_dh1 * d_h1_dw1
                self.w2 -= learn_rate * dl_dy_pred * dy_pred_dh1 * d_h1_dw2
                self.b1 -= learn_rate * dl_dy_pred * dy_pred_dh1 * d_h1_db1
                
                #Neuron 2
                self.w3 -= learn_rate * dl_dy_pred * dy_pred_dh2 * d_h2_dw3
                self.w4 -= learn_rate * dl_dy_pred * dy_pred_dh2 * d_h2_dw4
                self.b2 -= learn_rate * dl_dy_pred * dy_pred_dh2 * d_h2_db2
                
                #Output Neuron
                self.w5 -= learn_rate * dl_dy_pred * dy_pred_dw5
                self.w6 -= learn_rate * dl_dy_pred * dy_pred_dw6
                self.b3 -= learn_rate * dl_dy_pred * dy_pred_b3
                
                #Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_error(all_y_true, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))

#Define DataSet
data = np.array([[-2, -1]
                 [25, 6]
                 [17, 4]
                 [-15, -6]])

all_y_true = np.array([1,
                       0,
                       0,
                       1])

#Train the neural network
network = NeuralNetwork()
network.train(data, all_y_true)
            
                
                
                
    
