# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient with respect to the loss                       #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        N = X.shape[0]
        
        # Forward pass: Linear layer (no bias)
        scores = X.dot(self.weights['W1'])  # (N, num_classes)
        
        # Apply ReLU activation
        scores_relu = self.ReLU(scores)
        
        # Apply softmax
        probs = self.softmax(scores_relu)
        
        # Compute loss
        loss = self.cross_entropy_loss(probs, y)
        
        # Compute accuracy
        accuracy = self.compute_accuracy(probs, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        
        # Backward pass
        # Gradient of softmax and cross-entropy: dL/d(scores_relu)
        dscores_relu = probs.copy()
        dscores_relu[range(N), y] -= 1
        dscores_relu /= N
        
        # Gradient through ReLU
        dscores = dscores_relu * self.ReLU_dev(scores)
        
        # Gradient of W1: dL/dW1 = X^T @ dscores
        self.gradients['W1'] = X.T.dot(dscores)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


