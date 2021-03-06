import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    a = config.get('learning_rate')
    # print('v: ',v)

    next_w = None
    #############################################################################
    # TODO: Implement the momentum update formula as given in lecture. Store    #
    # the updated value in the next_w variable. You should also use and update  #
    # the velocity v.                                                           # 
    #############################################################################
    
    m = config.get('momentum')
    # print('w',w)
    # print('dw',a*dw)
    # print('m*v',a*m*v)
    v = (np.add(v,dw))/2
    next_w = w - a * (dw + m*v)
    # next_w =  m*v - dw 
    # print('next_w',next_w)
    # print('v: ',v)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    #############################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x   #
    # in the next_x variable. Don't forget to update cache value stored in      #
    # config['cache'] and to use the epsilon scalar to avoid dividing by zero.  #
    #############################################################################
    e = config.get('epsilon')
    lr = config.get('learning_rate')
    b = config.get('decay_rate')
    sk_1 = config.get('cache')
    sk = b*sk_1 + (1 - b)*(np.multiply(dx,dx))
    config['cache'] = sk
    next_x = x - lr * (dx)/(e + np.sqrt(sk))
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-10)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config and to use the epsilon scalar to avoid dividing by zero. #
    #############################################################################
    lr = config.get('learning_rate')
    b1 = config.get('beta1')
    b2 = config.get('beta2')
    e = config.get('epsilon')
    m_1 = config.get('m')
    v_1 = config.get('v')
    
    
    m = (1 - b1)*dx + b1*m_1
    config['m'] = m
    v = (1 - b2)*np.multiply(dx,dx) + b2*v_1
    config['v'] = v

    mh = m/(1 - b1)
    vh = v/(1 - b2)
    next_x = x - lr*mh/(np.sqrt(vh) + e)
    # print('x next',next_x)
    config['t'] += 1
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config
