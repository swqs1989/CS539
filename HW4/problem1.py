import torch as th
from torch.nn import Module, CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import SGD

#-------------------------------------------------------------------------
'''
    Problem 1: Softmax Regression using Pytorch 
    In this problem, you will implement the softmax regression method using PyTorch. 
    The main goal of this problem is to get familiar with the pytorch package for deep learning methods.
    You may need to use a Linux or Mac OS computer to install pytorch package.
    Please follow the instructions in this page to install pytorch package: http://pytorch.org/
    For windows computers, you could use dual OS, a virtual machine or docker to create a linux environment.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use torch.nn.Linear in the problem. Use Pytorch tensors/Variables to implement your own version of softmax regression.
'''

#-------------------------------------------------------
class SoftmaxRegression(Module):
    '''SoftmaxRegression is the softmax regression model with a single linear layer'''
    # ----------------------------------------------
    def __init__(self, p, c):
        ''' Initialize the softmax regression model. Create parameters W and b. Create a loss function object.  
            Inputs:
                p: the number of input features, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.W: the weight matrix parameter, a torch Variable of shape (p, c), initialized as all-zeros
                self.b: the bias vector parameter, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression.
            Hint: you could use CrossEntropyLoss class in torch.nn.
            Note: In this problem, the parameters are initialized as all-zeros for testing purpose only. In real-world cases, we usually initialize them with random values.
        '''
        super(SoftmaxRegression, self).__init__()
        #########################################
        ## INSERT YOUR CODE HERE


        #########################################

    # ----------------------------------------------
    def forward(self, x):
        '''
           Given a batch of training instances, compute the linear logits z. 
            Input:
                x: the feature vectors of a batch of training instance, a float torch Tensor of shape n by p. Here p is the number of features/dimensions. 
                    n is the number of instances in the batch.
                self.W: the weight matrix of softmax regression, a float torch Variable matrix of shape (p by c). Here c is the number of classes.
                self.b: the bias values of softmax regression, a float torch Variable vector of shape c by 1.
            Output:
                z: the logit values of the batch of training instances, a float matrix of shape n by c. Here c is the number of classes
            Hint: you could solve this problem using one line of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        #########################################
        return z

    #-----------------------------------------------------------------
    def compute_L(self, z,y):
        '''
            Compute multi-class cross entropy loss, which is the loss function of softmax regression. 
            Input:
                z: the logit values of training instances in a mini-batch, a float matrix of shape n by c. Here c is the number of classes
                y: the labels of a batch of training instances, an integer vector of length n. The values can be 0,1,2, ..., or (c-1).
            Output:
                L: the cross entropy loss of the batch, a float scalar. It's the average of the cross entropy loss on all instances in the batch.
            Hint: you could solve this problem using one line of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        #########################################
        return L 



    #-----------------------------------------------------------------
    def backward(self, L):
        '''
           Back Propagation: given compute the local gradients of the logits z, activations a, weights W and biases b on the instance. 
            Input:
                L: the cross entropy loss of the batch, a float scalar. It's the average loss on all instances in the batch.
            Output:
                self.W.grad: the average of the gradient of loss L w.r.t. the weight matrix W in the batch of training instances, a float matrix of shape (p by c). 
                       The i,j -th element of dL_dW represents the partial gradient of the loss w.r.t. the weight W[i,j]:   d_L / d_W[i,j]
                self.b.grad: the average of the gradient of the loss L w.r.t. the biases b, a float vector of length c . 
                       Each element dL_db[i] represents the partial gradient of loss L w.r.t. the i-th bias:  d_L / d_b[i]
            Hint: pytorch Variable provides automatic computation of gradients. You could solve this problem using one line of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        #########################################



    # ----------------------------------------------
    def train(self, loader, n_epoch=10,alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                n_epoch: the number of epochs, an integer scalar
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
              Note: after each training step, please set the gradients to 0 before starting the next training step.
        """

        # create a SGD optimizer
        optimizer = SGD([self.W,self.b], lr=alpha)
        # go through the dataset n_epoch times
        for _ in xrange(n_epoch):
            # use loader to load one batch of training data
            for x,y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                #########################################
                ## INSERT YOUR CODE HERE

                # forward pass

                # compute loss 

                # backward pass: compute gradients

                # update model parameters

                # reset the gradients of W and b to zero

                #########################################

    #--------------------------
    def test(self, loader):
        '''
           Predict the labels of one batch of testing instances using softmax regression.
            Input:
                loader: dataset loader, which loads one batch of dataset at a time.
            Output:
                accuracy: the accuracy 
        '''
        correct = 0.
        total = 0.
        # load dataset
        for x,y in loader:
            x = Variable(x) # one batch of testing instances, wrapped in Variable
            #########################################
            ## INSERT YOUR CODE HERE

            # predict labels of the batch of testing data


            #########################################
            total += y.size(0)
            correct += (y_predicted == y).sum()
        accuracy = correct / total
        return accuracy




