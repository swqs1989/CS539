import numpy as np
def compute_dL_da(a, y):
    '''
        Compute local gradient of the multi-class cross-entropy loss function w.r.t. the activations.
        Input:
            a: the activations of a training instance, a float numpy vector of shape c by 1. Here c is the number of classes.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        Output:
            dL_da: the local gradients of the loss function w.r.t. the activations, a float numpy vector of shape c by 1.
                   The i-th element dL_da[i] represents the partial gradient of the loss function w.r.t. the i-th activation a[i]:  d_L / d_a[i].
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dL_da = [0 for i in range(a.shape[0])]
    if a[y] != 0:
        dL_da[y] = -1. / a[y, 0]
    else:
        dL_da[y] = -1. / 999999.
    dL_da = np.matrix(dL_da).T
    #########################################
    return dL_da

def test_compute_dL_da():
    '''(1 point) compute_dL_da'''
    a  = np.mat('0.5;0.5')
    y = 1
    dL_da = compute_dL_da(a,y)

    assert type(dL_da) == np.matrixlib.defmatrix.matrix
    assert dL_da.shape == (2,1)
    assert np.allclose(dL_da, np.mat('0.;-2.'), atol= 1e-3)

    a  = np.mat('0.5;0.5')
    y = 0
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, np.mat('-2.;0.'), atol= 1e-3)

    a  = np.mat('0.1;0.6;0.1;0.2')
    y = 3
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, np.mat('0.;0.;0.;-5.'), atol= 1e-3)

    a  = np.mat('1.;0.')
    y = 1
    dL_da = compute_dL_da(a,y)

    np.seterr(all='raise')
    print dL_da[1] < -1e5
    print dL_da
    assert np.allclose(dL_da[0], 0., atol= 1e-3)
    assert dL_da[1] < -1e5
    assert dL_da[1] > -float('Inf')
    assert np.allclose(a, np.mat('1.;0.'))

test_compute_dL_da()