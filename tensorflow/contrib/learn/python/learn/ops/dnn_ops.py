"""TensorFlow ops for deep neural networks."""

from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.ops import dropout_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs


def dnn(tensor_in, hidden_units, activation=nn.relu, dropout=None):
  """Creates fully connected deep neural network subgraph.

  Args:
    tensor_in: tensor or placeholder for input features.
    hidden_units: list of counts of hidden units in each layer.
    activation: activation function between layers. Can be None.
    dropout: if not None, will add a dropout layer with given probability.

  Returns:
    A tensor which would be a deep neural network.
  """
  with vs.variable_scope('dnn'):
    for i, n_units in enumerate(hidden_units):
      with vs.variable_scope('layer%d' % i):
        # Weight initializer was set to None to replicate the behavior of
        # rnn_cell.linear. Using fully_connected's default initializer gets
        # slightly worse quality results on unit tests.
        tensor_in = layers.legacy_fully_connected(
            tensor_in,
            n_units,
            weight_init=None,
            weight_collections=['dnn_weights'],
            bias_collections=['dnn_biases'])
        if activation is not None:
          tensor_in = activation(tensor_in)
        if dropout is not None:
          is_training = array_ops_.squeeze(ops.get_collection('IS_TRAINING'))
          tensor_in = control_flow_ops.cond(
              is_training,
              lambda: dropout_ops.dropout(tensor_in, prob=(1.0 - dropout)),
              lambda: tensor_in)
    return tensor_in
