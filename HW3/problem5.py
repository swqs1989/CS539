import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        for val in np.unique(Y):
            prob = 1.0 * np.sum(D[Y == val])
            if prob != 0:
                e += -1.0 * prob / np.sum(D) * math.log(prob / np.sum(D), 2)
            else:
                e += 0
        #########################################
        return e 
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        groups = {}
        groupsd = {}
        ce = 0
        for x, y, d in zip(X, Y, D):
            try:
                groups[x].append(y)
                groupsd[x].append(d)
            except KeyError:
                groups[x] = [y]
                groupsd[x] = [d]
        for k,v in groups.iteritems():
            ce += (1.0 * np.sum(D[X == k])) \
                   * DS.entropy(np.array(v), np.array(groupsd[k]))
        #########################################
        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y, D) - DS.conditional_entropy(Y, X, D)
        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X, Y)
        th = -1
        g = -1
        try:
            for v in cp:
                XX = np.copy(X)
                XX = np.array(["T" if x > v else "F" for x in XX])
                ig = DS.information_gain(Y, XX, D)
                if ig > g:
                    th = v
                    g = ig
        except TypeError:
            return -float('Inf'), -1
        #########################################
        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        infos = []
        ths = []
        for j in range(X.shape[0]):
            tth, g = DS.best_threshold(X[j, :], Y, D)
            ths.append(tth)
            infos.append(g)
        i = np.argmax(infos)
        th = ths[i]
        #########################################
        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        max = 0
        y = None
        for val in np.unique(Y):
            v = np.sum(D[Y == val])
            if v > max:
                max = v
                y = val
        #########################################
        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X, Y, isleaf=False)
        t.p = DS.most_common(Y, D)
        # if Condition 1 or 2 holds, stop splitting 
        if DT.stop1(Y) or DT.stop2(X):
            t.isleaf = True
            return t

        # find the best attribute to split
        t.i, t.th = self.best_attribute(X, Y, D)

        # configure each child node
        t.C1 = Node(X[:, X[t.i, :] < t.th],
                    Y[X[t.i, :] < t.th],
                    isleaf=True,
                    p=DS.most_common(Y[X[t.i, :] < t.th], D[X[t.i, :] < t.th]))
        t.C2 = Node(X[:, X[t.i, :] >= t.th],
                    Y[X[t.i, :] >= t.th],
                    isleaf=True,
                    p=DS.most_common(Y[X[t.i, :] >= t.th], D[X[t.i, :] >= t.th]))
        #########################################
        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = float(np.sum(D[Y != Y_]))
        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        try:
            a = 0.5 * math.log((1 - e) / e)
        except ZeroDivisionError:
            a = 699.
        except ValueError:
            a = -699.
        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t, f = Y == Y_, Y != Y_
        D = np.copy(D)
        D[t] = D[t] * np.exp(-1.0 * a)
        D[f] = D[f] * np.exp(1.0 * a)
        D = D / float(np.sum(D))
        #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ds = DS()
        t = ds.build_tree(X, Y, D)
        if t.isleaf:
            Y_ = np.repeat(t.p, len(Y))
        else:
            Y_ = np.repeat(None, len(Y))
            Y_[X[t.i, :] < t.th] = t.C1.p
            Y_[X[t.i, :] >= t.th] = t.C2.p
        e = AB.weighted_error_rate(Y, Y_, D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D, a, Y, Y_)
        #########################################
        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y = []
        for t in T:
            if t.isleaf:
                p = t.p
            else:
                p = t.C1.p if x[t.i] < t.th else t.C2.p
            y.append(p)
        label = np.unique(y)
        y = np.array([1. if yy == label[0] else -1. for yy in y])
        y = label[0] if np.sum(y * A) > 0 else label[1]
        #########################################
        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for x in X.T:
            Y.append(AB.inference(x, T, A))
        Y = np.array(Y)
        #########################################
        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # initialize weight as 1/n
        D = 1. * np.ones(X.shape[1]) / X.shape[1]

        # iteratively build decision stumps
        T = []
        A = []
        for _ in range(n_tree):
            t, a, D = AB.step(X, Y, D)
            T.append(t)
            A.append(a)

        T = np.array(T)
        A = np.array(A)
        #########################################
        return T, A
   



 
