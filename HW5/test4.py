from problem4 import *
import gym
import sys
import numpy as np
from torch.autograd import Variable

'''
    Unit test 4:
    This file includes unit tests for problem4.py.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''
#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 4 (20 points in total)--------------'''
    assert sys.version_info[0]==2 # require python 2 (instead of python 3)

 
#-------------------------------------------------------------------------
def test_agent_init():
    '''agent_init (2 point)'''
    n = np.random.randint(2,100)
    d = np.random.randint(2,100)
    m = QNet(n,d) 
    assert type(m.W) == Variable 
    assert type(m.W.data) == th.FloatTensor
    assert m.W.requires_grad== True
    assert np.allclose(m.W.data.size(),(n,d))
    assert np.allclose(m.W.data, np.zeros((n,d))) 
    assert type(m.e) == float 
    assert np.allclose(m.e, 0.1) 
    assert type(m.n) == int 
    assert np.allclose(m.n, n) 
 
#-------------------------------------------------------------------------
def test_compute_Q():
    '''compute_Q(3 point)'''
    # 2 actions, 3 dimensional state
    m = QNet(2,3,0.) 
    m.W.data[1,1] +=1.
    s = Variable(th.Tensor([0.,1.,0.])) 
    Q = m.compute_Q(s)
    assert type(Q) == Variable 
    assert np.allclose(Q.data.size(),(2))
    assert np.allclose(Q.data, [0,1]) 

    m.W.data[1,0] +=2.
    s = Variable(th.Tensor([1.,0.,0.])) 
    Q = m.compute_Q(s)
    assert np.allclose(Q.data, [0,2]) 


 
#-------------------------------------------------------------------------
def test_agent_forward():
    '''agent_forward (5 point)'''
    m = QNet(2,3,0.) 
    m.W.data[1,1] +=1.
    s = Variable(th.Tensor([0.,1.,0.])) 
    a = m.forward(s)
    assert a == 1

    s = Variable(th.Tensor([1.,0.,0.])) 
    a = m.forward(s)
    assert a == 0

 
#-------------------------------------------------------------------------
def test_compute_L():
    '''compute_L(5 point)'''
    # 2 actions, 3 dimensional state
    m = QNet(2,3,0.) 
    s = Variable(th.Tensor([0.,1.,0.])) 
    s_new = Variable(th.Tensor([1.,0.,0.])) 
    L = m.compute_L(s=s,a=0,r=1.,s_new=s_new,gamma=1.)
    assert type(L) == Variable 
    assert np.allclose(L.data,[1])

    m.W.data[1,1] +=1.
    L = m.compute_L(s=s,a=1,r=1.,s_new=s_new,gamma=1.)
    assert np.allclose(L.data,[0])
    m.W.data[1,0] +=1.
    L = m.compute_L(s=s,a=1,r=1.,s_new=s_new,gamma=.5)
    assert np.allclose(L.data,[.25])
    
    # check gradient
    L.backward()
    print m.W.grad
    assert np.allclose(m.W.grad.data,[[0,0,0],[0,-1,0]])


#-------------------------------------------------------------------------
def test_play():
    '''agent_play (5 point)'''
    env = Game() 
    m = QNet(e=1.) 
    r=m.play(env,1000)
    print m.W.data
    print r
    assert r > 5 
    m.e = 0.1
    r=m.play(env,1000)
    print m.W.data
    print r
    assert r >= 200







