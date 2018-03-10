from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
#-------------------------------------------------------------------------
'''
    Problem 3: Recurrent Neural Network 
    In this problem, you will implement the recurrent neural network for sequence classification problems.
    We will use cross entropy as the loss function and stochastic gradient descent to train the model parameters.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
    Note: please do NOT use torch.nn.RNN, implement your own version of RNN using only basic tensor operations.
'''


#--------------------------
def tanh(z):
    """ Compute the hyperbolic tangent of the elements
        math: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        Input:
            z:  a batch of training instances, a float torch Variable of shape (n by p). n is the number instances in a batch. p is the number of features.
        Output:
            a: the nonlinear activation tensor, a float torch Variable of shape (n by n_filters by h by w )
        Note: please do NOT use th.nn.Tanh, th.tanh, th.nn.functional.tanh, implement your own version using basic tensor operations, such as exp(), div(), etc.
    """
    #########################################
    ## INSERT YOUR CODE HERE
    upbound = Variable(th.ones(z.size()) * 80.)
    lowbound = Variable(th.ones(z.size()) * -80.)

    z = th.min(z, upbound)
    z = th.max(z, lowbound)
    expz = th.exp(z)
    expz_ = th.exp(-1. * z)
    a = (expz - expz_) / (expz + expz_)

    #########################################
    return a


#-------------------------------------------------------
class RNN(sr):
    '''RNN is a recurrent neural network with hyperbolic tangent function as the activation function.
       After the recurrent layer, we apply a fully connected layer from hidden neurons to produce the output.
    '''
    # ----------------------------------------------
    def __init__(self, p, h=10, c=10):
        ''' Initialize the model. Create parameters of recurrent neural network. 
            Inputs:
                p: the number of input features, an integer scalar
                h: the number of memory (hidden) neurons, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.U: the weight matrix connecting the input features to the hidden units, a torch Variable of shape h by p, initialized as all-zeros. 
                self.V: the weight matrix connecting the hidden units to hidden units, a torch vector Variable of shape h by h, initialized as all-zeros.
                self.b_h: the bias vector of the hidden units, a torch vector Variable of length h, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer from hidden units to the output prediction, a torch Variable of shape (p, c), initialized as all-zeros. 
                        Hint: RNN is a subclass of SoftmaxRegression, which already has a W parameter and b. 
                self.b: the bias vector parameter of the outputs, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            You could solve this problem using 4 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        self.U = Variable(th.FloatTensor(p, h).zero_(), requires_grad=True)
        self.V = Variable(th.FloatTensor(h, h).zero_(), requires_grad=True)
        self.b_h = Variable(th.ones(h), requires_grad=True)
        super(RNN, self).__init__(h, c)
        #########################################


    # ----------------------------------------------
    def forward(self, x, H):
        '''
           Given a batch of training instances (with one time step), compute the linear logits z in the outputs.
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by p. Here n is the batch size. p is the number of features. 
                H:  the hidden state of the RNN model, a float torch Tensor of shape  n by h. Here h is the number of hidden units. 
            Output:
                z: the logit values of the batch of training instances after the output layer, a float matrix of shape n by c. Here c is the number of classes
                H_new: the new hidden state of the RNN model, a float torch Tensor of shape n by h.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        x = x.mm(self.U) + self.b_h.expand(x.size()[0], self.b_h.size()[0]) # n by h
        h = H.mm(self.V) # n by h
        z = x + h
        H_new = tanh(z) # n by h

        z = th.mm(H_new, self.W) + self.b.expand(x.size()[0], self.b.size()[0])
        #########################################
        return z, H_new


    # ----------------------------------------------
    def train(self, loader, n_steps=10,alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                        x: a batch of training instance, a float torch Tensor of shape n by t by p. Here n is the batch size. p is the number of features. t is the number of time steps.
                        y: a batch of training labels, a torch LongTensor of shape n by t. 
                n_steps: the number of batches of data to train, an integer scalar. Note: the n_steps is the number of training steps, not the number of time steps (t).
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
                Note: the loss of a sequence is computed as the sum of the losses in all time steps of the sequence.
        """
        # create a SGD optimizer
        optimizer = SGD([self.U,self.V,self.b_h,self.W,self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x,y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                n,t,p = x.size()
                h,_ = self.V.size()
                # initialize hidden state as all zeros
                H = Variable(th.zeros(n,h))
                #########################################
                ## INSERT YOUR CODE HERE
                # go through each time step
                loss = 0
                for num in range(t):
                    # forward pass
                    z, H_new = self.forward(x[:, num, :], H)
                    # compute loss
                    loss += self.compute_L(z, y[:, num])

                    H = H_new

                # backward pass: compute gradients
                self.backward(loss)
                # update model parameters
                self.U.data -= alpha * self.U.grad.data
                self.V.data -= alpha * self.V.grad.data
                self.b_h.data -= alpha * self.b_h.grad.data

                self.W.data -= alpha * self.W.grad.data
                self.b.data -= alpha * self.b.grad.data

                # reset the gradients to zero
                self.U.grad.data.zero_()
                self.V.grad.data.zero_()
                self.b_h.grad.data.zero_()

                self.W.grad.data.zero_()
                self.b.grad.data.zero_()

                #########################################
                count+=1
                if count >=n_steps:
                    return

