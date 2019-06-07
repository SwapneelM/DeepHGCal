import tensorflow as tf
import warnings

import argparse

import subprocess
import shutil
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.profiler import option_builder
import time
import sys

from tensorflow.python import debug as tf_debug

import numpy as np
import os
import configparser as cp

from tensorflow.python import debug as tf_debug

from queue import Queue
from threading import Thread, Lock
from os import path
import gzip
import pickle

import inspect
import importlib
from pdb import set_trace
import math

class ModelBuilder:
    '''
    Build the model based on the provided configuration
    '''
    def __init__(self, config):
        self.config = config

        # Add check for one_hot_labels - duplicated in TNTuplesClusteringTrainer
        # For God/Developer only knows what reason...
        # What was the point of reading in the config in the trainer if you re-read it here...?
        # TODO: Refactor to remove Model Builder
        try:
            self.one_hot_labels = self.config['one_hot_labels']
            print("One-Hot Labels set to ", self.one_hot_labels)
        except KeyError:
            self.one_hot_labels = False
        # Add condition for handling targets if one-hot-encoded labels are present
        if self.one_hot_labels == 'True':
            print("\nExtracting data dimensions containing one-hot labels...")
            # This should be a two-element array to be used as range of one-hot-labels in the data columns
            one_hot_dim_range = [int(x) for x in (self.config['target_indices']).split(',')]
            self.target_indices = tuple(range(one_hot_dim_range[0], one_hot_dim_range[1] + 1))
            print("\nNumber of one-hot-encoded labels present in data: ", len(self.target_indices))
        else:
            print("\nOne-Hot labels not present!")
            self.target_indices = tuple([int(x) for x in (self.config['target_indices']).split(',')])

        self.arguments_tuple = (
            len(tuple([int(x) for x in (self.config['input_spatial_features_indices']).split(',')])),
            len(tuple([int(x) for x in (self.config['input_spatial_features_local_indices']).split(',')])),
            len(tuple([int(x) for x in (self.config['input_other_features_indices']).split(',')])),
            len(self.target_indices),
            int(self.config['batch_size']),
            int(self.config['max_entries']),
            float(self.config['learning_rate']))

    def get_model(self):
        model_type = self.config['model_type']
        try:
            model = globals()[model_type](*self.arguments_tuple)
            print("Model Type:", type(model))
            return model
        except KeyError:
            raise RuntimeError("KeyError in ModelBuilder")


class TNTuplesEntriesReader():
    '''
    Read in the Data from the TFRecords
    This assumes the data has been written in a specific format with dimensions (num_max_entries, num_data_dims)
    '''
    def __init__(self, files_list, num_max_entries, num_data_dims, num_batch, repeat=True, shuffle_size=None):
        '''
        :param files_list:
        :param num_batch:
        :param num_max_entries:
        :param num_data_dims:
        :param repeat:
        :param shuffle_size:
        '''
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
        '''
        Returns the feeds (data, num_entries)
        '''
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


# Imported from DeepHGCal.python.lib.helpers
def get_num_parameters(scope=None):
    total_parameters = 0
    for variable in tf.trainable_variables(scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


# Imported from ops
def zero_out_by_energy(net):
    return tf.cast(tf.not_equal(net[..., 3], 0), tf.float32)[..., tf.newaxis] * net


def construct_sparse_io_dict(all_features, spatial_features_global, spatial_features_local, num_entries):
    '''
    Constructs dictionary for readers of sparse convolution layers

    :param all_features: All features tensor.  Should be of shape [batch_size, num_entries, num_features]
    :param spatial_features_global: Space like features tensor. Should be of shape [batch_size, num_entries, num_features]
    :param spatial_features_local: Space like features tensor (sensor sizes, etc.). Should be of shape [batch_size, num_entries, num_features]
    :param num_entries: Number of entries tensor for each batch entry.
    :return: dictionary in the format of the sparse conv layer
    '''
    return {
        'all_features': all_features,
        'spatial_features_global': spatial_features_global,
        'spatial_features_local': spatial_features_local,
        'num_entries': num_entries
    }


def sparse_conv_collapse(sparse_dict):
    # This is how PEP8's E128 requires formatting to be done
    # We would do well to adhere to best practices in our code
    all_features, spatial_features_global, \
    spatial_features_local, num_entries = \
        sparse_dict['all_features'], sparse_dict['spatial_features_global'], \
        sparse_dict['spatial_features_local'], sparse_dict['num_entries']
    return tf.concat([spatial_features_global, all_features, spatial_features_local], axis=-1)


def high_dim_dense(inputs, nodes, **kwargs):
    if len(inputs.shape) == 3:
        return tf.layers.conv1d(inputs, nodes, kernel_size=(1), strides=(1), padding='valid',
                                **kwargs)
    if len(inputs.shape) == 4:
        return tf.layers.conv2d(inputs, nodes, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                **kwargs)
    if len(inputs.shape) == 5:
        return tf.layers.conv3d(inputs, nodes, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
                                **kwargs)


def gauss(x):
    return tf.exp(-1 * x * x)


def apply_edges(vertices, edges, reduce_sum=True, flatten=True,expand_first_vertex_dim=True, aggregation_function=tf.reduce_max):
    '''
    edges are naturally BxVxV'xF
    vertices are BxVxF'  or BxV'xF'
    This function returns BxVxF'' if flattened and summed
    '''
    edges = tf.expand_dims(edges,axis=3)
    if expand_first_vertex_dim:
        vertices = tf.expand_dims(vertices,axis=1)
    vertices = tf.expand_dims(vertices,axis=4)

    out = edges*vertices # [BxVxV'x1xF] x [Bx1xV'xF'x1] = [BxVxV'xFxF']

    if reduce_sum:
        out = aggregation_function(out,axis=2)
    if flatten:
        out = tf.reshape(out,shape=[out.shape[0],out.shape[1],-1])

    return out


def sparse_conv_global_exchange(vertices_in,
                                aggregate_function=tf.reduce_mean,
                                expand_to_dims=-1,
                                collapse_to_dims=-1,
                                learn_global_node_placement_dimensions=None):

    trans_vertices_in = vertices_in
    if expand_to_dims > 0:
        trans_vertices_in = tf.layers.dense(trans_vertices_in, expand_to_dims, activation=tf.nn.relu)

    if learn_global_node_placement_dimensions is not None:
        trans_vertices_in_space = trans_vertices_in[:, :, 0:learn_global_node_placement_dimensions]
        global_node_placement = tf.reduce_mean(trans_vertices_in_space, axis=1, keepdims=True)
        edges = create_edges(trans_vertices_in_space, global_node_placement,
                             norotation=True)
        edges = gauss_times_linear(edges)
        global_summed = apply_edges(trans_vertices_in, edges, reduce_sum=True, flatten=True)

    else:
        global_summed = tf.reduce_mean(trans_vertices_in, axis=1, keepdims=True)

    global_summed = tf.tile(global_summed, [1, vertices_in.shape[1], 1])
    vertices_out = tf.concat([vertices_in, global_summed], axis=-1)
    if collapse_to_dims > 0:
        vertices_out = tf.layers.dense(vertices_out, collapse_to_dims, activation=tf.nn.relu)

    return vertices_out


def sparse_conv_hidden_aggregators(vertices_in,
                                   n_aggregators,
                                   n_filters,
                                   pre_filters=[],
                                   n_propagate=-1,
                                   plus_mean=False,
                                   return_agg=False
                                   ):
    vertices_in_orig = vertices_in
    trans_vertices = vertices_in
    for f in pre_filters:
        trans_vertices = tf.layers.dense(trans_vertices, f, activation=tf.nn.relu)

    if n_propagate > 0:
        vertices_in = tf.layers.dense(vertices_in, n_propagate, activation=None)

    agg_nodes = tf.layers.dense(trans_vertices, n_aggregators, activation=None)  # BxVxNA, vertices_in: BxVxF
    agg_nodes = gauss(agg_nodes)
    vertices_in = tf.concat([vertices_in, agg_nodes], axis=-1)

    edges = tf.expand_dims(agg_nodes, axis=3)  # BxVxNAx1
    edges = tf.transpose(edges, perm=[0, 2, 1, 3])  # [BxVxV'xF]

    print('Edges:', edges.shape)
    print('Vertices_in:', vertices_in.shape)

    vertices_in_collapsed = apply_edges(vertices_in, edges, reduce_sum=True,
                                        flatten=True)  # , aggregation_function=tf.reduce_mean)# [BxNAxF]
    vertices_in_mean_collapsed = apply_edges(vertices_in, edges, reduce_sum=True,
                                             flatten=True, aggregation_function=tf.reduce_mean)  # [BxNAxF]
    print('Vertices_in_collapsed:', vertices_in_collapsed.shape)

    edges = tf.transpose(edges, perm=[0, 2, 1, 3])  # [BxVxV'xF]

    expanded_collapsed = apply_edges(vertices_in_collapsed, edges, reduce_sum=False, flatten=True)  # [BxVxF]
    print('Expanded_collapsed:', expanded_collapsed.shape)
    expanded_collapsed = tf.concat([vertices_in_orig, expanded_collapsed, agg_nodes], axis=-1)
    print('Expanded_collapsed2:', expanded_collapsed.shape)

    merged_out = high_dim_dense(expanded_collapsed, n_filters, activation=tf.nn.tanh)
    return merged_out, agg_nodes


# The class that handles training, testing, and plotting for the model
class TNTuplesClusteringTrainer:
    def read_config(self, config_file_path, config_name):
        config_file = cp.ConfigParser()
        config_file.read(config_file_path)
        self.config = config_file[config_name]

    def __init__(self, config_file, config_name):
        self.read_config(config_file, config_name)
        self.config_name = config_name

        self.from_scratch = int(self.config['from_scratch']) == 1
        self.model_path = self.config['model_path']
        self.summary_path = self.config['summary_path']
        self.test_out_path = self.config['test_out_path']
        self.profile_out_path = self.config['profiler_out_path']
        self.train_for_iterations = int(self.config['train_for_iterations'])
        self.save_after_iterations = int(self.config['save_after_iterations'])
        self.learning_rate = float(self.config['learning_rate'])
        self.training_files = self.config['training_files_list']
        self.validation_files = self.config['validation_files_list']
        self.test_files = self.config['test_files_list']

        self.validate_after = int(self.config['validate_after'])
        self.num_testing_samples = int(self.config['num_testing_samples'])

        self.num_batch = int(self.config['batch_size'])
        self.num_max_entries = int(self.config['max_entries'])
        self.num_data_dims = int(self.config['num_data_dims'])

        # Add check for one_hot_labels
        try:
            self.one_hot_labels = self.config['one_hot_labels']
            print("One-Hot Labels set to ", self.one_hot_labels)
        except KeyError:
            self.one_hot_labels = False
        try:
            self.output_seed_indices = int(self.config['output_seed_indices_in_inference'])==1
        except KeyError:
            self.output_seed_indices = False

        # Delete plotting code and set plot_after to -1
        self.plot_after = -1

        self.spatial_features_indices = tuple([int(x) for x in (self.config['input_spatial_features_indices']).split(',')])
        self.spatial_features_local_indices = tuple([int(x) for x in (self.config['input_spatial_features_local_indices']).split(',')])
        self.other_features_indices = tuple([int(x) for x in (self.config['input_other_features_indices']).split(',')])

        # Add condition for handling targets if one-hot-encoded labels are present
        if self.one_hot_labels == 'True':
            print("\nExtracting data dimensions containing one-hot labels...")
            # This should be a two-element array to be used as range of one-hot-labels in the data columns
            one_hot_dim_range = [int(x) for x in (self.config['target_indices']).split(',')]
            self.target_indices = tuple(range(one_hot_dim_range[0], one_hot_dim_range[1] + 1))
            print("\nNumber of one-hot-encoded labels present in data: ", len(self.target_indices))
        else:
            print("\nOne-Hot labels not present!")
            self.target_indices = tuple([int(x) for x in (self.config['target_indices']).split(',')])
        self.reader_type = self.config['reader_type'] if len(self.config['reader_type']) != 0 else \
            "data_and_num_entries_reader"

        # self.reader_factory = ReaderFactory()
        self.model = None

    def initialize(self):
        self.model = ModelBuilder(self.config).get_model()
        self.model.config_name = self.config_name
        try:
            self.model.set_training(True)
        except AttributeError:
            pass
        self.model.initialize()
        self.saver_sparse = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model.get_variable_scope())
        )

    def initialize_test(self):
        self.model = ModelBuilder(self.config).get_model()
        self.model.config_name = self.config_name
        try:
            self.model.set_training(False)
        except AttributeError:
            pass

        self.model.initialize()
        self.saver_sparse = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model.get_variable_scope())
        )

    def clean_summary_dir(self):
        print("Cleaning summary dir")
        for the_file in os.listdir(self.summary_path):
            file_path = os.path.join(self.summary_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def train(self):
        self.initialize()
        print("Beginning to train network with parameters", get_num_parameters(self.model.get_variable_scope()))
        print("Variable scope:", self.model.get_variable_scope())
        placeholders = self.model.get_placeholders()

        if self.from_scratch:
            subprocess.call("mkdir -p %s"%(self.summary_path), shell=True)
            subprocess.call("mkdir -p %s"%(self.test_out_path), shell=True)
            subprocess.call("mkdir -p %s"%(os.path.join(self.test_out_path, 'ops')), shell=True)
            with open(self.model_path + '_code.py', 'w') as f:
                f.write(self.model.get_code())

            '''
            ops_parent = os.path.dirname(ops.__file__)
            for ops_file in os.listdir(ops_parent):
                if not ops_file.endswith('.py'):
                    continue
                shutil.copy(os.path.join(ops_parent, ops_file), os.path.join(self.test_out_path, 'ops'))
            '''
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_output = self.model.get_compute_graphs()
        graph_temp = self.model.get_temp()

        if self.plot_after != -1:
            data_plotting = None  # TODO: Load

        if self.from_scratch:
            self.clean_summary_dir()

        inputs_feed = TNTuplesEntriesReader(self.training_files, self.num_max_entries, self.num_data_dims, self.num_batch).get_feeds()
        inputs_validation_feed = TNTuplesEntriesReader(self.validation_files, self.num_max_entries, self.num_data_dims, self.num_batch).get_feeds(shuffle=False)
        print("\nFeed Input shape: ", inputs_feed[0].get_shape().as_list())

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        with tf.Session() as sess:
            # Allow access to the Tensorflow Debugger CLI
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

            if not self.from_scratch:
                self.saver_sparse.restore(sess, self.model_path)
                print("\n\nINFO: Loading model\n\n")
                with open(self.model_path + '.txt', 'r') as f:
                    iteration_number = int(f.read())
            else:
                iteration_number = 0

            print("Starting iterations")
            while iteration_number < self.train_for_iterations:
                inputs_train = sess.run(list(inputs_feed))

                print("Input Train shape: ", inputs_train[0].shape)

                learning_rate = 1
                if hasattr(self.model, "learningrate_scheduler"):
                    learning_rate = self.model.learningrate_scheduler.get_lr(iteration_number)
                else:
                    learning_rate = self.model.learning_rate
                if iteration_number == 0:
                    print('learning rate ', learning_rate)

                if len(placeholders) == 5:
                    inputs_train_dict = {
                        placeholders[0]: inputs_train[0][:, :, self.spatial_features_indices],
                        placeholders[1]: inputs_train[0][:, :, self.spatial_features_local_indices],
                        placeholders[2]: inputs_train[0][:, :, self.other_features_indices],
                        placeholders[3]: inputs_train[0][:, :, self.target_indices],
                        placeholders[4]: inputs_train[1],
                        self.model.is_train: True,
                        self.model.learning_rate: learning_rate
                    }
                else:
                    raise ValueError("Length of Placeholders is more than inputs_train_dict requires. \
                        This might be due to old code.")

                t, eval_loss, _, eval_summary, eval_output = sess.run([graph_temp, graph_loss, graph_optmiser, graph_summary, graph_output], feed_dict=inputs_train_dict)

                if self.plot_after != -1:
                    if iteration_number % self.plot_after == 0:
                        # TODO: Add code to automate plots for validation here
                        pass

                if iteration_number % self.validate_after == 0:
                    inputs_validation = sess.run(list(inputs_validation_feed))
                    self.inputs_plot = inputs_validation

                    if len(placeholders) == 5:
                        inputs_validation_dict = {
                            placeholders[0]: inputs_validation[0][:, :, self.spatial_features_indices],
                            placeholders[1]: inputs_validation[0][:, :, self.spatial_features_local_indices],
                            placeholders[2]: inputs_validation[0][:, :, self.other_features_indices],
                            placeholders[3]: inputs_validation[0][:, :, self.target_indices],
                            placeholders[4]: inputs_validation[1],
                            self.model.is_train: False,
                            self.model.learning_rate: learning_rate
                        }
                    else:
                        raise ValueError("Length of Placeholders is more than inputs_train_dict requires. \
                        This might be due to old code.")

                    eval_loss_validation, eval_summary_validation = sess.run([graph_loss, graph_summary_validation],
                                                                             feed_dict=inputs_validation_dict)
                    summary_writer.add_summary(eval_summary_validation, iteration_number)
                    print("Validation - Iteration %4d: loss %.6E" % (iteration_number, eval_loss_validation))

                print("Training - Iteration %4d: loss %0.6E" % (iteration_number, eval_loss))
                print(t[0])
                iteration_number += 1
                summary_writer.add_summary(eval_summary, iteration_number)
                if iteration_number % self.save_after_iterations == 0:
                    print("\n\nINFO: Saving model\n\n")
                    self.saver_sparse.save(sess, self.model_path)
                    with open(self.model_path + '.txt', 'w') as f:
                        f.write(str(iteration_number))

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)

    def visualize(self):
        self.initialize_test()
        print("Beginning to visualize network with ", get_num_parameters(self.model.get_variable_scope()), " parameters")
        placeholders = self. model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_output = self.model.get_compute_graphs()
        graph_temp = self.model.get_temp()
        layer_feats = self.model.temp_feat_visualize

        inputs_feed = TNTuplesEntriesReader(self.test_files, self.num_max_entries, self.num_data_dims, self.num_batch).get_feeds(shuffle=False)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            self.saver_sparse.restore(sess, self.model_path)
            print("\n\nINFO: Loading model", self.model_path,"\n\n")

            print("Starting visualizing")
            iteration_number = 0
            while iteration_number < int(np.ceil(self.num_testing_samples / self.num_batch)):
                inputs_test = sess.run(list(inputs_feed))
                print("Run")

                if len(placeholders) == 5:
                    inputs_train_dict = {
                        placeholders[0]: inputs_test[0][:, :, self.spatial_features_indices],
                        placeholders[1]: inputs_test[0][:, :, self.spatial_features_local_indices],
                        placeholders[2]: inputs_test[0][:, :, self.other_features_indices],
                        placeholders[3]: inputs_test[0][:, :, self.target_indices],
                        placeholders[4]: inputs_test[1],
                        self.model.is_train: False,
                        self.model.learning_rate: 0
                    }
                else:
                    raise NotImplementedError("Old code is being called. Placeholder size needs to be checked.")

                eval_out = sess.run([graph_temp, graph_loss, graph_output], feed_dict=inputs_train_dict)
                # layer_outs = eval_out[3:]

                # Get the maximum of the predicted values of the tracks
                # argmax will return the index of the track it is most likely to be associated with
                prediction = tf.argmax(eval_out[2], axis=2)
                original_label = tf.argmax(inputs_test[0][:, :, self.target_indices], axis=2)

                set_trace()

                # Put the condition here!
                iteration_number += 1

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)

    def test(self):
        self.initialize_test()
        print("Beginning to test network with parameters", get_num_parameters(self.model.get_variable_scope()))
        placeholders = self. model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_output = self.model.get_compute_graphs()
        graph_temp = self.model.get_temp()

        inputs_feed = TNTuplesEntriesReader(self.test_files, self.num_max_entries, self.num_data_dims, self.num_batch).get_feeds(shuffle=False)
        inference_streamer = InferenceOutputStreamer(output_path=self.test_out_path, cache_size=100)
        inference_streamer.start_thread()
        print(type(inputs_feed))
        print("****************************************")
        print("Test Input shape: ", inputs_feed[0].get_shape().as_list())
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            self.saver_sparse.restore(sess, self.model_path)
            print("\n\nINFO: Loading model", self.model_path,"\n\n")

            print("Starting testing")
            iteration_number = 0
            while iteration_number < int(np.ceil(self.num_testing_samples / self.num_batch)):
                inputs_test = sess.run(list(inputs_feed))

                if len(placeholders) == 5:
                    inputs_train_dict = {
                        placeholders[0]: inputs_test[0][:, :, self.spatial_features_indices],
                        placeholders[1]: inputs_test[0][:, :, self.spatial_features_local_indices],
                        placeholders[2]: inputs_test[0][:, :, self.other_features_indices],
                        placeholders[3]: inputs_test[0][:, :, self.target_indices],
                        placeholders[4]: inputs_test[1],
                        self.model.is_train: False,
                        self.model.learning_rate: 0
                    }
                else:
                    raise ValueError("Length of Placeholders is more than inputs_train_dict requires. \
                    This might be due to old code.")

                t, eval_loss, eval_output = sess.run([graph_temp, graph_loss, graph_output], feed_dict=inputs_train_dict)

                print("Adding", len(inputs_test[0]), "test results")
                for i in range(len(inputs_test[0])):
                    if not self.output_seed_indices:
                        inference_streamer.add((inputs_test[0][i], (inputs_test[1])[i,0], eval_output[i]))
                    else:
                        inference_streamer.add((inputs_test[0][i], (inputs_test[1])[i,0], eval_output[i], inputs_test[2][i]))

                print("Testing - Sample %4d: loss %0.5f" % (iteration_number*self.num_batch, eval_loss))
                print(t[0])
                iteration_number += 1

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)

        inference_streamer.close()


class lr_scheduler(object):
    def __init__(self, lr_dict=[], lr=0.001):

        self.lr_dict = lr_dict
        self.lr = lr
        self.next_change = 0

    def check_next(self, iteration):

        for i in range(len(self.lr_dict)):
            key = self.lr_dict[i][0]
            self.next_change = key
            if key > iteration:
                print('next change at iteration ', key, 'just changed learning rate to ', self.lr)
                return
            self.lr = self.lr_dict[i][1]
        print('iteration', iteration , 'learning rate at ', self.lr)

    def get_lr(self,iteration):
        if iteration >= self.next_change:
            self.check_next(iteration)
        return self.lr

    def create_exponential_wiggle(self, start_lr, end_lr, end_exp_iterations, wiggle_frequency=0.2, n_points=500, scaler=1000):
        effective_constant_x = 6./float(float(end_exp_iterations)/float(scaler))
        self.lr_dict = []
        for i in range(n_points):
            lr = start_lr * math.exp(- float(i) * effective_constant_x) * (1. + 0.5 * math.cos(wiggle_frequency*float(i))) \
                 + end_lr *(1. + 0.1 * math.cos(wiggle_frequency*float(i)))
            self.lr_dict.append((int(scaler*i) , lr))
        print(self.lr_dict)


class SparseConvClusteringSpatialMinLoss2:
    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        self.weight_weights = []
        self.AdMat = None
        self.use_seeds = True
        self.mean_sqrt_resolution = None
        self.variance_sqrt_resolution = None
        self.total_loss = None
        self.fixed_seeds = None
        self.momentum = 0.6
        self.varscope = 'sparse_conv_clustering_spatial1'
        self.freeze_bn_after = None
        self.E_loss = False
        self.sum_loss = True
        self.log_loss = False
        self.initialized = False
        self.n_space = n_space
        self.n_space_local = n_space_local
        self.n_other_features = n_others
        self.n_target_dim = n_target_dim
        self.batch_size = batch_size
        self.max_entries = max_entries
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.start_learning_rate = learning_rate
        self.learningrate_scheduler = lr_scheduler(lr=learning_rate)
        self.use_seeds = False
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def initialize(self):
        if self.initialized:
            print("Already initialized")
            return
        self._construct_graphs()

    def get_summary(self):
        return self._graph_summaries

    def get_performance_measures(self):
        raise RuntimeError("Unimplemented")

    def get_summary_validation(self):
        return self._graph_summaries_validation

    def get_placeholders_dict(self):
        placeholders = dict()
        placeholders['features_space'] = self._placeholder_space_features
        placeholders['features_space_local'] = self._placeholder_space_features_local
        placeholders['features_others'] = self._placeholder_other_features
        placeholders['targets'] = self._placeholder_targets
        placeholders['num_entries'] = self._placeholder_num_entries
        return placeholders

    def get_compute_graphs(self):
        return self._graph_output

    def get_losses(self):
        return self._graph_loss

    def get_optimizer(self):
        return self._graph_optimizer

    def get_temp(self):
        return self._graph_temp

    def get_code(self):
        return inspect.getsource(importlib.import_module(self.__module__))

    def _get_loss(self):
        diff_sq = (self._graph_output - self._placeholder_targets) ** 2 * tf.cast(
            tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1),
                             maxlen=self.max_entries)[:, :, tf.newaxis], tf.float32)
        diff_sq = tf.reduce_sum(diff_sq, axis=[-1, -2])

        loss_unreduced = (diff_sq / tf.cast(self._placeholder_num_entries, tf.float32)) * tf.cast(
            self._placeholder_num_entries != 0, tf.float32)

        return tf.reduce_mean(loss_unreduced)

    def make_placeholders(self):
        print("Making placeholders")
        self._placeholder_space_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])
        self._placeholder_space_features_local = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space_local])
        self._placeholder_other_features = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_other_features])
        self._placeholder_targets = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_target_dim])
        self._placeholder_num_entries = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, 1])

    def get_placeholders(self):
        return self._placeholder_space_features, self._placeholder_space_features_local, self._placeholder_other_features, \
               self._placeholder_targets, self._placeholder_num_entries  # , self._placeholder_seed_indices

    def get_variable_scope(self):
        return self.config_name

    def tntuples_loss(self):
        assert self._graph_output.shape[2] >= 2
        print('TNTuples Loss\n===========\n')

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        print('num_entries: ', num_entries.shape)

        prediction = self._graph_output
        targets = self._placeholder_targets
        print('prediction:', prediction.shape)
        print('targets:', targets.shape)

        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=targets)
        return tf.reduce_mean(cross_entropy_loss)

    def _get_loss(self):
        return self.tntuples_loss()

    def compute_output_only_global_exchange(self,_input,seed_idxs):

        feat = sparse_conv_collapse(_input)
        feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum,center=False)
        global_feat = []
        depth = 13
        for i in range(depth):
            feat = sparse_conv_global_exchange(feat,
                                               expand_to_dims=-1,
                                               collapse_to_dims=42,
                                               learn_global_node_placement_dimensions=3)

            feat = tf.layers.batch_normalization(feat,training=self.is_train, momentum=self.momentum,center=False)
            print('feat '+str(i), feat.shape)
            if i%2 or i==depth-1:
                global_feat.append(feat)

        feat = tf.concat(global_feat,axis=-1)
        print('feat concat', feat.shape)
        feat = tf.layers.dense(feat,32, activation=tf.nn.relu)
        feat = tf.layers.dense(feat,3, activation=tf.nn.relu)

        return feat

    def compute_output_hidden_aggregators(self, _input, seeds, plusmean=False):
        feat = sparse_conv_collapse(_input)
        feat = zero_out_by_energy(feat)
        feat = tf.layers.batch_normalization(feat, training=self.is_train, momentum=self.momentum)

        aggregators = 11*[4]
        filters     = 11*[32]
        propagate   = 11*[20]
        pre_filters = 11*[[]]

        feat = sparse_conv_global_exchange(feat)
        feat = tf.layers.batch_normalization(feat, training=self.is_train, momentum=self.momentum)
        feat = high_dim_dense(feat, 32, activation=tf.nn.tanh)
        feat_list = []
        self.temp_feat_visualize = []
        for i in range(len(filters)):
            feat, _ = sparse_conv_hidden_aggregators(feat,
                                                     aggregators[i],
                                                     n_filters=filters[i],
                                                     pre_filters=pre_filters[i],
                                                     n_propagate=propagate[i],
                                                     plus_mean=plusmean
                                                     )
            self.temp_feat_visualize.append(_)
            feat = tf.layers.batch_normalization(feat, training=self.is_train, momentum=self.momentum)
            feat_list.append(feat)
            # feat = tf.layers.dropout(feat, rate=0.0005, training=self.is_train)

        feat = tf.concat(feat_list, axis=-1)
        feat = tf.layers.dense(feat, 100, activation=tf.nn.relu)
        # feat = tf.layers.dense(feat, 3, activation=tf.nn.relu)

        return feat

    def _compute_output(self):

        feat = self._placeholder_other_features
        print("\nFeatures:", feat.shape)
        space_feat = self._placeholder_space_features
        local_space_feat = self._placeholder_space_features_local
        num_entries = self._placeholder_num_entries
        n_batch = space_feat.shape[0]

        # For Jan: Are these the random seeds to be modified to tracks? Would that help with better/faster clustering?
        nrandom = 1
        random_seeds = tf.random_uniform(shape=(int(n_batch), nrandom), minval=0, maxval=3600, dtype=tf.int64)
        # print('random_seeds', random_seeds.shape)
        # TODO: For future reference, this is where to modify the code for adding custom seeds/aggregators
        try:
            seeds = tf.concat([self._placeholder_seed_indices, random_seeds], axis=-1)
            seeds = tf.transpose(seeds, [1, 0])
            seeds = tf.random_shuffle(seeds)
            seeds = tf.transpose(seeds, [1, 0])
            seeds = self._placeholder_seed_indices
            print('seeds', seeds.shape)
        except AttributeError:
            print("\nSetting 'seeds' to None since _placeholder_seed_indices is not defined")
            seeds = None
            pass

        net_in = construct_sparse_io_dict(feat, space_feat, local_space_feat,
                                          tf.squeeze(num_entries))

        # Exclude normalisation for tntuples
        # Declare 'net' as the same variable 'net_in' to maintain variable names
        net = net_in
        self.sum_loss = True
        output = self.compute_output_hidden_aggregators(net, seeds, plusmean=True)

        output = tf.layers.dense(output, 100)
        print("\nOutput:", output.shape)
        self._graph_temp = tf.nn.softmax(output)
        self._graph_temp = tf.reduce_sum(self._graph_temp[:, :, :], axis=1)/2679.
        return output

    def _construct_graphs(self):
        print("Constructing Graphs in Inherited SparseConvClustering Model")
        self.learningrate_scheduler.create_exponential_wiggle(self.start_learning_rate,
                                                              self.start_learning_rate/80,
                                                              end_exp_iterations=2000000,
                                                              scaler=25000,
                                                              wiggle_frequency=0.1,
                                                              n_points=50)
        with tf.variable_scope(self.get_variable_scope()):
            self.initialized = True
            self.weight_init_width = 1e-6

            print("\nMaking placeholders (make_placeholders).....")
            self.make_placeholders()

            print("\nComputing output (_compute_output).....")
            self._graph_output = self._compute_output()
            print("\nGraph Output:", self._graph_output.shape)

            print("\nGetting loss (_get_loss).....")
            self._graph_loss = self._get_loss()
            print("\nGraph Loss:", self._graph_loss.shape)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print("\nUpdate ops:", update_ops)
            with tf.control_dependencies(update_ops):
                self._graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._graph_loss)

            # Repeating, maybe there is a better way?
            self._graph_summary_loss = tf.summary.scalar('loss', self._graph_loss)
            self._graph_summaries = tf.summary.merge([self._graph_summary_loss,
                                                      # tf.summary.scalar('mean-res', self.mean_resolution),
                                                      # tf.summary.scalar('variance-res', self.variance_resolution),
                                                      # tf.summary.scalar('mean-res-sqrt', self.mean_sqrt_resolution),
                                                      # tf.summary.scalar('variance-res-sqrt', self.variance_sqrt_resolution),
                                                      tf.summary.scalar('learning-rate', self.learning_rate)])

            self._graph_summary_loss_validation = tf.summary.scalar('Validation loss', self._graph_loss)
            self._graph_summaries_validation = tf.summary.merge([self._graph_summary_loss_validation])

    def get_losses(self):
        print("Hello, world!")
        return self._graph_loss


# Parse arguments from the command
parser = argparse.ArgumentParser(description='Run training for graph based clustering')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--test', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--profile', default=False, help="Whether to run evaluation on test set")
parser.add_argument('--visualize', default=False, help="Whether to run layer wise visualization (x-mode only)")
args = parser.parse_args()

trainer = TNTuplesClusteringTrainer(args.input, args.config)

# Test and train are the only necessary options
if args.test:
    trainer.test()
elif args.visualize:
    trainer.visualize()
else:
    trainer.train()
