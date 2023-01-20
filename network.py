import numpy as np

class MLP(object):
    
    def __init__(self):
        self._input_layer = np.zeros((784, 1))
        self._hidden_layer_1 = np.zeros((64, 1))
        self._hidden_layer_2 = np.zeros((32, 1))
        self._output_layer = np.zeros((10, 1))
        
        self._weights_ih = np.random.uniform(-1, 1, (64, 784))
        self._biases_ih = np.zeros((64, 1))
        
        self._weights_hh = np.random.uniform(-1, 1, (32, 64))
        self._biases_hh = np.zeros((32, 1))
        
        self._weights_ho = np.random.uniform(-1, 1, (10, 32))
        self._biases_ho = np.zeros((10, 1))
           
    def predict(self, image):
        self._forward_propagation(image)
        return np.argmax(self._output_layer)
    
    def fit(self, epochs, learning_rate, training_images, training_labels):
        for epoch in range(epochs):
            error = 0
            prediction_count = 0 
            for training_image, training_label in zip(training_images, training_labels):
                y_hat = self.predict(training_image) 
                y = np.argmax(training_label) 
                
                error += self._cost(training_label, self._output_layer)
                
                if y_hat == y:
                    prediction_count += 1
                
                self._back_propagation(training_label, learning_rate)
                
            error = round(error / training_images.shape[0], 5)
            acc = round((prediction_count / training_images.shape[0]) * 100, 2)
            print(f"Epoch: {epoch+1} - Accuracy: {acc}%, Error: {error}")
        
    def _forward_propagation(self, image):
        self._input_layer = image
        self._hidden_layer_1 = self._sigmoid(np.dot(self._weights_ih, self._input_layer) + self._biases_ih)
        self._hidden_layer_2 = self._sigmoid(np.dot(self._weights_hh, self._hidden_layer_1) + self._biases_hh)
        self._output_layer = self._sigmoid(np.dot(self._weights_ho, self._hidden_layer_2) + self._biases_ho)
         
    def _back_propagation(self, label, learning_rate):
        self._delta_output_layer = self._cost_derivative(label, self._output_layer) * self._sigmoid_derivative(self._output_layer)
        self._delta_hidden_layer_2 = self._sigmoid_derivative(self._hidden_layer_2) * np.dot(np.transpose(self._weights_ho), self._delta_output_layer)
        self._delta_hidden_layer_1 = self._sigmoid_derivative(self._hidden_layer_1) * np.dot(np.transpose(self._weights_hh), self._delta_hidden_layer_2)
        
        self._weights_ih += (-1) * learning_rate * np.dot(self._delta_hidden_layer_1, np.transpose(self._input_layer))
        self._biases_ih += (-1) * learning_rate * self._delta_hidden_layer_1
        
        self._weights_hh += (-1) * learning_rate * np.dot(self._delta_hidden_layer_2, np.transpose(self._hidden_layer_1))
        self._biases_hh += (-1) * learning_rate * self._delta_hidden_layer_2
        
        self._weights_ho += (-1) * learning_rate * np.dot(self._delta_output_layer, np.transpose(self._hidden_layer_2))
        self._biases_ho += (-1) * learning_rate * self._delta_output_layer

    def _cost(self, y, y_hat):
        return (1/2) * np.sum(pow(y - y_hat, 2))
    
    def _cost_derivative(self, y, y_hat):
        return y_hat - y
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)