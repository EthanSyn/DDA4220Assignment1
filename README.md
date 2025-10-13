# Coding Exercise

# Overview

Deep Neural Networks are becoming more and more popular and widely applied to many ML-related domains. In this assignment, you will complete a simple pipeline of training neural networks to recognize MNIST Handwritten Digits: http://yann.learcun.com/exdb/mnist/. You'll implement two neural network architectures along with the code to load data, train and optimize these networks. You will also run different experiments on your model to complete a short report. Be sure to use the template of report we give to you and fill in your information on the first page.

The main.py contains the major logic of this assignment. You can execute it by invoking the following command where the yaml file contains all the hyper-parameters.

$ python main.py --config config=<name_of_config_file>.yaml

There are three pre-defined config files under ./configs. Two of them are default hyperparameters for models that you will implement in the assignment (Softmax Regression and 2-layer MLP). The correctness of your implementation is partially judged by the model performance on these default hyper-parameters; therefore, do NOT modify values in these config files. The third config file, config_exp.yaml, is used for your hyper-parameter tuning experiments (details in Section 5) and you are free to modify values of the hyper-parameters in this file.

The script trains a model with the number of epochs specified in the config file. At the end of each epoch, the script evaluates the model on the validation set. After the training completes, the script finally evaluates the best model on the test data.

# Python and dependencies

In this assignment, we will work with Python 3. If you do not have a python distribution installed yet, we recommend installing Anaconda: https://www.anaconda.com/ (or miniconda) with Python 3. We provide environment.yaml which contains a list of libraries needed to set environment for this assignment. You can use it to create a copy of conda environment:

$ conda env create -f environment.yaml

If you already have your own Python development environment, please refer to this file to find necessary libraries, which is used to set the same coding/grading environment.

# Code Test

Some public unit tests are provided in the tests/ in the assignment repository. You can test each part of your implementation with these test cases by:

$ python -m unitittest tests.<name_of/tests>

However, passing all local tests neither means your code is free of bugs nor guarantees that you will receive full credits for the coding section.

# 1 Data Loading

Data loading is the very first step of any machine learning pipelines. First, you should download the MNIST dataset with our provided script under ./data by:

$ cd data

$ sh get_data.sh

$ cd ..

# Microsoft Windows 10 Only

C:\assignmentfolder> cd data

C:\assignmentfolder\data> get_data.bat

C:\assignmentfolder\data> cd ..

The script downloads MNIST data (mnist_train.csv and mnist_test.csv) to the ./data folder.

# 1.1 Data Preparation

To avoid the choice of hyper-parameters overfits the training data, it is a common practice to split the training dataset into the actual training data and validation data and perform hyper-parameter tuning based on results on validation data. Additionally, in deep learning, training data is often forwarded to models in batches for faster training time and noise reduction.

In our pipeline, we first load the entire MNIST data into the system, followed by a training/validation split on the training set. We simply use the first  $80\%$  of the training set as our training data and use the rest training set as our validation data. We also want to organize our data (training, validation, and test) in batches and use different combination of batches in different epochs for training data. Therefore, your tasks are as follows:

1. follow the instruction in code to complete load_mnist_trainval in ./utils.py for training/validation split  
2. follow the instruction in code to complete generate_batched_data in ./utils.py to organize data in batches

You can test your data loading code by running:

$ python -m unitittest tests.test_loading

# 2 Model Implementation

You will now implement two networks from scratch: a simple softmax regression and a two-layer multi-layer perceptron (MLP). Definitions of these classes can be found in ./models.

Weights of each model will be randomly initialized upon construction and stored in a weight dictionary. Meanwhile, a corresponding gradient dictionary is also created and initialized to zeros. Each model only has one public method called forward, which takes input of batched data and corresponding labels and returns the loss and accuracy of the batch. Meanwhile, it computes gradients of all weights of the model (even though the method is called forward!) based on the training batch.

# 2.1 Utility Function

There are a few useful methods defined in ./base_network.py that can be shared by both models. Your first task is to implement them based on instructions in _base_network.py:

1. Activation Functions. There are two activation functions needed for this assignment: ReLU and Sigmoid. Implement both functions as well as their derivatives in ./base_network.py (i.e., sigmoid, sigmoid_dev, ReLU, and ReLU_dev). Test your methods with:

$ python -m unitittest tests.testactivation

2. Loss Functions. The loss function used in this assignment is Cross Entropy Loss. You will need to implement both Softmax function and the computation of Cross Entropy Loss in ./base_network.py. HINT: You may want to checkout the numerically stable version of softmax: https://deepnotes.io/softmax-crossentropy. Test (this test also tests the accuracy method):

$ python -m unitittest tests.test_loss

3. Accuracy. We are also interested in knowing how our model is doing on a given batch of data. Therefore, you may want to implement the compute.accuracy method in ./base_network.py to compute the accuracy of a given batch.

# 2.2 Model Implementation

You will implement the training processes of a simple Softmax Regression and a two-layer MLP in this section. The Softmax Regression is composed by a fully-connected layer followed by a ReLU activation. The two-layer MLP is composed by two fully-connected layers with a Sigmoid Activation in between. Note that the Softmax Regression model has no bias terms, while the two-layer MLP model does use biases. Also, don't forget the softmax function before computing your loss!

1. Implement the forward method in softmax_regression.py as well as two_layer_nn.py. If the mode argument is train, compute gradients of weights and store the gradients in the gradient dictionary. Otherwise, simply return the loss and accuracy. Test:

$ python -m unitittest tests.test_network

# 3 Optimizer

We will use an optimizer to update weights of models. An optimizer is initialized with a specific learning rate and a regularization coefficient. Before updating model weights, the optimizer applies L2 regularization on the model:

$$
J = L _ {C E} + \frac {1}{2} \lambda \sum_ {i = 1} ^ {N} w _ {i} ^ {2}
$$

where  $J$  is the overall loss and  $L_{CE}$  is the Cross-Entropy loss computed between predictions and labels.

You will also implement a vanilla SGD optimizer. The update rule is as follows:

$$
\theta^ {t + 1} = \theta^ {t} - \eta \nabla_ {\theta} J (\theta)
$$

where  $\theta$  is the model parameter,  $\eta$  stands for learning rate and the  $\nabla$  term corresponds to the gradient of the parameter.

In summary, your tasks are as follows:

1. Follow instructions in the code and implement applyRegularExpression in_base_OPTimizer.py. Remember, you may NOT want to apply regularization on bias terms!  
2. Implement the update method in sgd.py based on the discussion of the update rule above.

Test your optimizer by running:

$ python -m unitittest tests.test_training

# 4 Visualization

It is always a good practice to monitor the training process by monitoring the learning curves. Our training method in main.py stores averaged loss and accuracy of the model on both training and validation data at the end of each epoch. Your task is to plot the learning curves by leveraging these values. A sample plot of learning curves can be found in Figure ??

1. Implement plot_curves in ./utils.py. You'll get full marks on this question as long as your plot makes sense.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-13/74bfe459-f94e-441c-bd79-34355d40d8f8/d23a84abe7bd45894ce625cdb58de893dbd394b3aa1635f19f8fe6dbc8246e26.jpg)  
Figure 1: Example loss curve

# 5 Experiments

Now, you have completed the entire training process. It's time to play with your model a little. You will use your implementation of the two-layer MLP for this section. There are different combinations of your hyper-parameters specified in the report template and your tasks are to tune those parameters and report your observations by answering questions in the report template. We provide a default config file config_exp.yaml in ./configs. When tuning a specific hyper-parameter (e.g., the learning rate), please leave all other hyper-parameters as-is in the default config file.

1. You will try out different values of learning rates and report your observations in the report file.  
2. You will try out different values of regularization coefficients and report your observations in the report file.  
3. You will try your best to tune the hyper-parameters for best accuracy.

# 6 Deliverables

# 6.1 Coding

You will need to submit a zip file containing all your codes in structure. For your convenience, we provide a handy script for you.

Simply run

$ bash collect_submission.sh

or if running Microsoft Windows 10

C:\assignmentfolder>collect Submission.bat

# 6.2 Writeup

You will also need to submit a report summarizing your experimental results and findings as specified in Section 5. Again, we provide a starting template for you and your task is just to answer each question in the template. For whichever questions asking for plots, please include plots from all your experiments.

Note: Explanations should go into why things work the way they do with proper deep learning principles and analysis. For example, with hyperparameter tuning you should explain the reasoning behind your choices and what behavior you expected. If you need more than one slide for a question, you are free to create new slides right after the given one.