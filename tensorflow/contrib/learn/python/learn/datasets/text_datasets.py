"""Text datasets."""

import os
import tarfile

import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.datasets import base

DBPEDIA_URL = 'https://googledrive.com/host/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M/dbpedia_csv.tar.gz'


def get_dbpedia(data_dir):
    train_path = os.path.join(data_dir, 'dbpedia_csv/train.csv')
    test_path = os.path.join(data_dir, 'dbpedia_csv/test.csv')
    if not (gfile.Exists(train_path) and gfile.Exists(test_path)):
        archive_path = base.maybe_download('dbpedia_csv.tar.gz', data_dir, DBPEDIA_URL)
        tfile = tarfile.open(archive_path, 'r:*')
        tfile.extractall(data_dir)
    train = base.load_csv(train_path, np.int32, 0, has_header=False)
    test = base.load_csv(test_path, np.int32, 0, has_header=False)
    datasets = base.Datasets(train=train, validation=None, test=test)
    return datasets


def load_dbpedia():
    return get_dbpedia('dbpedia_data')

