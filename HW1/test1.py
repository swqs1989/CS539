from problem1 import *
import numpy as np
import sys
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (15 points in total)---------------------'''
    assert sys.version_info[0]==2 # require python 2 (instead of python 3)


#-------------------------------------------------------------------------
def test_compute_Phi():
    ''' (5 points) compute_Phi'''
    x = np.mat('1.;2.;3')

    Phi = compute_Phi(x,2) 
    assert type(Phi) == np.matrixlib.defmatrix.matrix 
    assert np.allclose(Phi.T, [[1,1,1],[1,2,3]], atol = 1e-3) 

    Phi = compute_Phi(x,3) 
    assert np.allclose(Phi.T, [[1,1,1],[1,2,3],[1,4,9]], atol = 1e-3) 

    Phi = compute_Phi(x,4) 
    assert np.allclose(Phi.T, [[1,1,1],[1,2,3],[1,4,9],[1,8,27]], atol = 1e-3) 



#-------------------------------------------------------------------------
def test_least_square():
    ''' (5 points) least square'''
    Phi = np.mat([[1.,1.,1.],[-1.,0.,1.]]).T
    y = np.mat('1.5;2.5;3.5')
    w = least_square(Phi,y)
    assert type(w) == np.matrixlib.defmatrix.matrix 
    assert np.allclose(w, np.mat('2.5;1.'), atol = 1e-2) 

    for _ in xrange(20):
        p = np.random.randint(2,8)
        print 'p:',p
        n = np.random.randint(200,400)
        print 'n:',n
        w_true = np.asmatrix(np.random.random(p)).T
        print 'w_true:',w_true
        x = np.asmatrix(np.random.random(n)*10).T
        Phi = compute_Phi(x,p)
        print 'Phi.shape:',Phi.shape
        e = np.asmatrix(np.random.randn(n)).T*0.01
        y = Phi*w_true + e
        print 'y.shape:',y.shape 
        w = least_square(Phi,y)
        print 'w:',w
        assert np.allclose(w,w_true, atol = 0.1)


#-------------------------------------------------------------------------
def test_l2_least_square():
    ''' (5 points) ridge regression'''
    Phi = np.mat([[1.,1.,1.],[-1.,0.,1.]]).T
    y = np.mat('1.5;2.5;3.5')
    w = ridge_regression(Phi,y)
    assert type(w) == np.matrixlib.defmatrix.matrix 
    assert np.allclose(w, np.mat('2.5;1.'), atol = 1e-2) 

    w = ridge_regression(Phi,y,alpha = 1000)
    assert np.allclose(w, np.mat('0.;0.'), atol = 1e-2) 

    for _ in xrange(20):
        p = np.random.randint(2,8)
        print 'p:',p
        n = np.random.randint(200,400)
        print 'n:',n
        w_true = np.asmatrix(np.random.random(p)).T
        print 'w_true:',w_true
        x = np.asmatrix(np.random.random(n)*10).T
        Phi = compute_Phi(x,p)
        print 'Phi.shape:',Phi.shape
        e = np.asmatrix(np.random.randn(n)).T*0.01
        y = Phi*w_true + e
        print 'y.shape:',y.shape 
        w = ridge_regression(Phi,y)
        print 'w:',w
        assert np.allclose(w,w_true, atol = 0.1)



