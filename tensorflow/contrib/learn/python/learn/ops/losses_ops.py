"""TensorFlow Ops for loss computation."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.contrib.losses.python.losses import loss_ops


def mean_squared_error_regressor(tensor_in, labels, weights, biases, name=None):
  """Returns prediction and loss for mean squared error regression."""
  with ops.op_scope([tensor_in, labels], name, "mean_squared_error_regressor"):
    predictions = nn.xw_plus_b(tensor_in, weights, biases)
    if len(labels.get_shape()) == 1:
      labels = array_ops_.reshape(labels, [-1, 1])
    return predictions, loss_ops.sum_of_squares(predictions, labels)


def softmax_classifier(tensor_in,
                       labels,
                       weights,
                       biases,
                       class_weight=None,
                       name=None):
  """Returns prediction and loss for softmax classifier.

  Args:
    tensor_in: Input tensor, [batch_size, feature_size], features.
    labels: Tensor, [batch_size, n_classes], labels of the output classes.
    weights: Tensor, [batch_size, feature_size], linear transformation
      matrix.
    biases: Tensor, [batch_size], biases.
    class_weight: Tensor, optional, [n_classes], weight for each class.
      If not given, all classes are supposed to have weight one.

  Returns:
    Prediction and loss tensors.
  """
  with ops.op_scope([tensor_in, labels], name, "softmax_classifier"):
    logits = nn.xw_plus_b(tensor_in, weights, biases)
    if class_weight is not None:
      logits = math_ops.mul(logits, class_weight)
    return nn.softmax(logits), loss_ops.softmax_cross_entropy(logits, labels)
