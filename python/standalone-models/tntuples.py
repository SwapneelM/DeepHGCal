import tensorflow as tf
import warnings

import argparse

import subprocess
import shutil
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.profiler import option_builder
import time
import sys

from inference import InferenceOutputStreamer
from pdb import set_trace
from tensorflow.python import debug as tf_debug


class TNTuplesEntriesReader():
    def __init__(self, files_list, num_max_entries, num_data_dims, num_batch, repeat=True, shuffle_size=None):
        self.files_list = files_list
        self.num_max_entries = num_max_entries
        self.num_data_dims = num_data_dims
        self.repeat = repeat
        self.num_batch = num_batch
        self.shuffle_size = shuffle_size
        self.return_seeds = False

    def _parse_function(self, example_proto):
        keys_to_features = {
            'data': tf.FixedLenFeature((self.num_max_entries, self.num_data_dims), tf.float32),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['data']

    def get_feeds(self, shuffle=True):
        """
        Returns the feeds (data, num_entries)

        :param files_list:
        :param num_batch:
        :param num_max_entries:
        :param num_data_dims:
        :param repeat:
        :param shuffle_size:
        :return:
        """
        with open(self.files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        dataset = dataset.map(self._parse_function)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_batch * 3 if self.shuffle_size is None else self.shuffle_size)
        dataset = dataset.repeat(None if self.repeat else 1)
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()

        num_entries = tf.ones(shape=(self.num_batch, 1), dtype=tf.int64) * self.num_max_entries

        if self.return_seeds:
            return data, num_entries
        else:
            return data[:, :, :], num_entries
