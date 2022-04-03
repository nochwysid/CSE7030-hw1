import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.weight_scale = weight_scale
        self.reg = reg
        '''just to keep 'solver' from exploding'''
        self.grads = {}

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        #pass
        weights1 = np.random.normal(0, weight_scale,(input_dim, hidden_dim))
        bias1 = np.zeros(hidden_dim)
        
        weights2 = np.random.normal(0,weight_scale,(hidden_dim, num_classes))
        bias2 = np.zeros(num_classes)
        
        self.params['W1']=weights1
        self.params['b1']=bias1
        self.params['W2']=weights2
        self.params['b2']=bias2
        # print('weights 1 shape:',weights1.shape)
        # print('weights 2 shape:',weights2.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        # print('\t model regularization factor:',self.reg)
        ''' * * * WHAT THE ACTUAL!! * * * '''
        # print('before:{}, after:{}'.format(np.sum(self.params['W1']),np.sum(self.params['W1'] + self.reg * np.sum(
                                          # np.dot(
                                          #   self.params['W1'],
                                          #   self.params['W1'].T
                                          #   )
                                          # ))))
        s2 = self.weight_scale
        #only for numerical gradient check
        # s2 = 0.68241372367235
        self.params['W1'] = self.params['W1'] - s2*0.5*self.reg * np.sum(
                                          np.dot(
                                            self.params['W1'],
                                            self.params['W1'].T
                                            )
                                          )
        self.params['W2'] = self.params['W2'] - s2*0.5*self.reg * np.sum(
                                          np.dot(
                                            self.params['W2'],
                                            self.params['W2'].T
                                            )
                                          )
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        out1, cache1 = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        out2, cache2 = affine_forward(out1, self.params['W2'],self.params['b2'])
        scores = out2
        #for i in range(X.shape[0]):
        #  scores[i] = np.argmax(out2)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
          scores = out2
          
          return scores

        ''' * * * * * * * * * * * * * * * '''
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization on the weights,    #
        # but not the biases.                                                      #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # s2 = self.weight_scale*self.weight_scale
        
        # print('reg',self.reg)
        loss, dx = softmax_loss(out2, y)
        
        ''' this should be the correct version '''
        layer2_L2 = self.reg *  0.5 * s2 * np.sum(
                                          np.dot(
                                            self.params['W2'],
                                            self.params['W2'].T
                                            )
                                          )
        layer1_L2 = self.reg *  0.5 * s2 * np.sum(
                                          np.dot(
                                            self.params['W1'],
                                            self.params['W1'].T
                                            )
                                          )
        
        
        grads = {}
        
        dx2, dw2, db2 = affine_backward(dx,cache2)
        grads['W2']=s2*(dw2 - layer2_L2)
        grads['b2']=s2*db2
        
        dx1, dw1, db1 = affine_relu_backward(dx2, cache1)
        grads['W1']=s2*(dw1 - layer1_L2)
        grads['b1']=s2*db1
        
        #print('avg dw2',np.average(grads['W2']),'avg dw1',np.average(grads['W1']))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_dropout = dropout > 0
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.weight_scale = weight_scale
        self.dtype = dtype
        self.num_layers = len(hidden_dims)
        self.seed = seed
        self.params = {}
        self.grads = {}


        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        print('number of layers:',self.num_layers)
        self.params['W1'] =  np.random.normal(0, 
                                              weight_scale,
                                                (self.input_dim,
                                                self.hidden_dims[0] 
                                                )
                                              )
        self.params['b1'] = np.zeros(hidden_dims[0])

        for i in range( self.num_layers-1):
          self.params['W{}'.format(i+2)] =  np.random.normal(0, 
                                                           weight_scale,
                                                            (self.hidden_dims[i],
                                                             self.hidden_dims[i+1]
                                                            )
                                                           )
          self.params['b{}'.format(i+2)] = np.zeros(hidden_dims[i+1])
        #print(self.params)
        # print(self.params['W1'].shape,self.params['W2'].shape,)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train',
                               'gamma{}'.format(i): 1.0,
                               'beta{}'.format(i): 0.0} 
                              for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype (causes 'AttributeError')
        # for k, v in self.params.items():
        #     print('type of v:',type(v))
        #     self.params[k] = v.astype(dtype)
        #     print('type after:',type(v))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        #print('\t model regularization factor:',self.reg)
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        '''forward'''
        ll = self.num_layers
        outlist = []
        cachelist = []
        out, cache = affine_relu_forward(X, 
                                      self.params['W1'],
                                      self.params['b1'])
        outlist.append(out)
        cachelist.append(cache)
        for i in range(1, ll):#needs to be i-2 and i +1
          # print('W{} shape:{}'.format(i+1,self.params['W{}'.format(i+1)].shape))
          out2, cache2 = affine_relu_forward(outlist[i - 1], 
                                        self.params['W{}'.format(i+1)],
                                        self.params['b{}'.format(i+1)])
          outlist.append(out2)
          cachelist.append(cache2)
        
        #scores = np.zeros((self.input_dim,self.num_classes))
        scores = outlist[len(outlist)-1]
        # print('len outlist',len(outlist))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        # loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization on the         #
        # weights, but not the biases.                                             #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        s2 = self.weight_scale
        loss, dx = softmax_loss(scores, y)
        
        co = self.reg * 0.5 * s2
        layerLast_L2 = []
        for i in range(ll):
          layerLast_L2.append( co * np.sum(
                                      np.dot(self.params['W{}'.format(ll-i)],
                                             self.params['W{}'.format(ll-i)].T
                                             ))) 
          
        backlist = []
        
        grads = {}
        dx, dw, db  = affine_relu_backward(dx, cachelist[ll - 1])
        backlist.append([dx, dw, db ])
        self.grads['W{}'.format(ll)] = s2*(dw - layerLast_L2[ll-1])
        self.grads['b{}'.format(ll)] = s2*(db)
        for i in range(1, ll):
          dx, dw, db  = affine_relu_backward(backlist[i-1][0], cachelist[ll - i - 1])
          backlist.append([dx, dw, db ])
          self.grads['W{}'.format(ll - i)] = s2*(dw - layerLast_L2[ll-i-1])
          self.grads['b{}'.format(ll - i)] = s2*(db)
        #print('grads keys',self.grads.keys())
        grads = self.grads
        # print('len backlist',len(backlist))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
