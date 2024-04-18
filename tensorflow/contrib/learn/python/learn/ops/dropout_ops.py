"""Dropout operations and handling."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs

# Key to collect dropout probabilities.
DROPOUTS = "dropouts"


def dropout(tensor_in, prob, name=None):
  """Adds dropout node and stores probability tensor into graph collection.

  Args:
    tensor_in: Input tensor.
    prob: Float or Tensor.

  Returns:
    Tensor of the same shape of `tensor_in`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
  with ops.op_scope([tensor_in], name, "dropout") as name:
    if isinstance(prob, float):
      prob = vs.get_variable("prob", [],
                             initializer=init_ops.constant_initializer(prob),
                             trainable=False)
    ops.add_to_collection(DROPOUTS, prob)
    return nn.dropout(tensor_in, prob)
