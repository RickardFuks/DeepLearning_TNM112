import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x #Return same value 
    elif activation == "relu":
        return np.maximum(0,x) #return 0 if negative otherwise the value of input thus (-) -> 0
    elif activation == "sigmoid":
        return 1/(1 + np.exp(-x))
    elif activation == "softmax":
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W) -1

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = 0
        for i in range(len(W)):
            self.N += np.size(W[i]) + np.size(b[i]) #calculate numbe rof trainable data by adding weights and biass

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self, 
        x
    ):
    
        # Store predictions
        y = np.zeros((len(x), self.dataset.K))
    
        # Loop through data points
        for i in range(len(x)):
            a = x[i].reshape(2, 1) #Make column vector
    
            # Hidden layers
            for j in range(self.hidden_layers):
                z = self.W[j] @ a + self.b[j] #weight sum + bias
                a = activation(z, self.activation) # use activation and send forward to a
    
            # Output layer
            z = self.W[-1] @ a + self.b[-1] #Last layer
            a = activation(z.T, "softmax") #Transpose that matrix! softmax makes it %prediction
    
            # Store prediction
            y[i] = a[0]
    
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        train_prediction = self.feedforward(self.dataset.x_train)
        train_loss = np.mean((train_prediction - self.dataset.y_train_oh) ** 2)
        train_acc = np.mean(np.argmax(train_prediction, axis=1) == self.dataset.y_train) * 100 #argmax find index of highest value
    
        print("\tTrain loss:     %0.4f" % train_loss)
        print("\tTrain accuracy: %0.2f" % train_acc)
    
        # Test data
        test_prediction = self.feedforward(self.dataset.x_test)
    
        test_loss = np.mean((test_prediction - self.dataset.y_test_oh) ** 2)
        test_acc = np.mean(np.argmax(test_prediction, axis=1) == self.dataset.y_test) * 100
    
        print("\tTest loss:      %0.4f" % test_loss)
        print("\tTest accuracy:  %0.2f" % test_acc)
