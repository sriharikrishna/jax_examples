# Initial state: x(0): 0
# Final state: x(100): 2000
# Transfer function: x(i+1) = x(i)+u_k(i)
# Constraint: x(i) <= 20

import jax.numpy as np
from jax import grad

from jax import vmap # for auto-vectorizing functions
from functools import partial # for use with vmap
from jax import jit # for compiling functions for speedup
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax # neural network layers
import matplotlib.pyplot as plt # visualization

from jax import random

import numpy as onp
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)

def cost(params, x, target):
  return target-sum(np.where(params <= 20.0, params, -params))+x

@jit
def step(i, opt_state, x, target):
    print("In step")
    p = get_params(opt_state)
    g = grad(cost)(p, x,target)
    return opt_update(i, g, opt_state)


def grape():
  step_count = 100
  rng = random.PRNGKey(0)
  u_k = random.normal(rng, (step_count,))
  print(u_k)

  opt_state = opt_init(u_k)

  xinput = 0
  target = 2000.0
  max_iterations=300
  predictions = [0 for x in range(300)]
  myiteration = 0
  while True:
    opt_state = step(myiteration, opt_state, xinput, target)
    predictions[myiteration] = sum(get_params(opt_state))
    if (abs(predictions[myiteration]-target)/target <0.00001):
      p = get_params(opt_state)
      print("Finally")
      print(opt_state)
      p = get_params(opt_state)
      print(sum(p))
      xrange_inputs = np.linspace(1,myiteration,myiteration).reshape((myiteration, 1)) # (k, 1)
      plt.plot(xrange_inputs, predictions[0:myiteration], label='prediction')
      plt.show()
      break
    myiteration = myiteration+1
    if(myiteration==max_iterations):
      break
    

grape()
