from ._base_optimizer import _BaseOptimizer
import numpy as np

class AdamW(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate, reg)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}   # first moment estimates
        self.v = {}   # second moment estimates
        self.t = 0    # timestep counter

    def apply_regularization(self, model):
        pass ## implement regularization in update loops

    def update(self, model):
        '''
        Update model weights using Adam optimizer
        :param model: The model with parameters and gradients
        :return: None, but the model weights should be updated
        '''
        self.t += 1

        #############################################################################
        # TODO:                                                                     #
        #    1) Initialize m and v for each parameter if not present                #
        #    2) Update biased first and second moment estimates                     #
        #    3) Compute bias-corrected estimates                                    #
        #    4) Update parameters using AdamW rule (including regularization)       #
        #############################################################################
        
        for key in model.weights:
            # Initialize m and v if not present
            if key not in self.m:
                self.m[key] = np.zeros_like(model.weights[key])
                self.v[key] = np.zeros_like(model.weights[key])
            
            # Get current gradient
            grad = model.gradients[key]
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # AdamW update rule: apply weight decay directly to weights
            if key.startswith('W'):  # Apply regularization only to weights, not biases
                model.weights[key] = model.weights[key] * (1 - self.learning_rate * self.reg)
            
            # Adam update
            model.weights[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
