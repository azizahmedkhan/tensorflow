"""Tests for learn.io.graph_io."""

import random

import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile

_VALID_FILE_PATTERN = "VALID"
_FILE_NAMES = [b"abc", b"def", b"ghi", b"jkl"]
_INVALID_FILE_PATTERN = "INVALID"


class GraphIOTest(tf.test.TestCase):

  def _mock_glob(self, pattern):
    if _VALID_FILE_PATTERN == pattern:
      return _FILE_NAMES
    self.assertEqual(_INVALID_FILE_PATTERN, pattern)
    return []

  def setUp(self):
    super(GraphIOTest, self).setUp()
    random.seed(42)
    self._orig_glob = gfile.Glob
    gfile.Glob = self._mock_glob

  def tearDown(self):
    gfile.Glob = self._orig_glob
    super(GraphIOTest, self).tearDown()

  def test_dequeue_batch_value_errors(self):
    default_batch_size = 17
    queue_capacity = 1234
    num_threads = 3
    name = "my_batch"

    self.assertRaisesRegexp(
        ValueError, "No files match",
        tf.contrib.learn.io.read_batch_features,
        _INVALID_FILE_PATTERN, default_batch_size, None, tf.TFRecordReader,
        False, queue_capacity,
        num_threads, name)
    self.assertRaisesRegexp(
        ValueError, "Invalid batch_size",
        tf.contrib.learn.io.read_batch_features,
        _VALID_FILE_PATTERN, None, None, tf.TFRecordReader,
        False, queue_capacity, num_threads, name)
    self.assertRaisesRegexp(
        ValueError, "Invalid batch_size",
        tf.contrib.learn.io.read_batch_features,
        _VALID_FILE_PATTERN, -1, None, tf.TFRecordReader,
        False, queue_capacity, num_threads, name)
    self.assertRaisesRegexp(
        ValueError, "Invalid queue_capacity",
        tf.contrib.learn.io.read_batch_features,
        _VALID_FILE_PATTERN, default_batch_size, None, tf.TFRecordReader,
        False, None, num_threads, name)
    self.assertRaisesRegexp(
        ValueError, "Invalid num_threads",
        tf.contrib.learn.io.read_batch_features,
        _VALID_FILE_PATTERN, default_batch_size, None, tf.TFRecordReader,
        False, queue_capacity, None,
        name)
    self.assertRaisesRegexp(
        ValueError, "Invalid num_threads",
        tf.contrib.learn.io.read_batch_features,
        _VALID_FILE_PATTERN, default_batch_size, None, tf.TFRecordReader,
        False, queue_capacity, -1,
        name)
    self.assertRaisesRegexp(
        ValueError, "Invalid batch_size",
        tf.contrib.learn.io.read_batch_features,
        _VALID_FILE_PATTERN, queue_capacity + 1, None, tf.TFRecordReader,
        False, queue_capacity, 1, name)

  def test_batch_tf_record(self):
    batch_size = 17
    queue_capacity = 1234
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      inputs = tf.contrib.learn.io.read_batch_examples(
          _VALID_FILE_PATTERN, batch_size,
          reader=tf.TFRecordReader, randomize_input=False,
          queue_capacity=queue_capacity, name=name)
      self.assertEquals("%s:0" % name, inputs.name)
      file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/fifo_queue" % name
      op_nodes = test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TFRecordReader" % name: "TFRecordReader",
          example_queue_name: "FIFOQueue",
          name: "QueueDequeueMany"
      }, g)
      self.assertAllEqual(_FILE_NAMES, sess.run(["%s:0" % file_names_name])[0])
      self.assertEqual(
          queue_capacity, op_nodes[example_queue_name].attr["capacity"].i)

  def test_batch_randomized(self):
    batch_size = 17
    queue_capacity = 1234
    name = "my_batch"

    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      inputs = tf.contrib.learn.io.read_batch_examples(
          _VALID_FILE_PATTERN, batch_size,
          reader=tf.TFRecordReader, randomize_input=True,
          queue_capacity=queue_capacity, name=name)
      self.assertEquals("%s:0" % name, inputs.name)
      file_name_queue_name = "%s/file_name_queue" % name
      file_names_name = "%s/input" % file_name_queue_name
      example_queue_name = "%s/random_shuffle_queue" % name
      op_nodes = test_util.assert_ops_in_graph({
          file_names_name: "Const",
          file_name_queue_name: "FIFOQueue",
          "%s/read/TFRecordReader" % name: "TFRecordReader",
          example_queue_name: "RandomShuffleQueue",
          name: "QueueDequeueMany"
      }, g)
      self.assertEqual(
          set(_FILE_NAMES), set(sess.run(["%s:0" % file_names_name])[0]))
      self.assertEqual(
          queue_capacity, op_nodes[example_queue_name].attr["capacity"].i)


if __name__ == "__main__":
  tf.test.main()
