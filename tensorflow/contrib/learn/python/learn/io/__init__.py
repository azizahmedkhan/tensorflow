"""Tools to allow different io formats."""

from tensorflow.contrib.learn.python.learn.io.dask_io import extract_dask_data
from tensorflow.contrib.learn.python.learn.io.dask_io import extract_dask_labels
from tensorflow.contrib.learn.python.learn.io.dask_io import HAS_DASK
from tensorflow.contrib.learn.python.learn.io.graph_io import read_batch_examples
from tensorflow.contrib.learn.python.learn.io.graph_io import read_batch_features
from tensorflow.contrib.learn.python.learn.io.graph_io import read_batch_record_features
from tensorflow.contrib.learn.python.learn.io.pandas_io import extract_pandas_data
from tensorflow.contrib.learn.python.learn.io.pandas_io import extract_pandas_labels
from tensorflow.contrib.learn.python.learn.io.pandas_io import extract_pandas_matrix
from tensorflow.contrib.learn.python.learn.io.pandas_io import HAS_PANDAS

# pylint: disable=g-import-not-at-top
if HAS_PANDAS:
  from tensorflow.contrib.learn.python.learn.io.pandas_io import pd
