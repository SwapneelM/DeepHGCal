import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv import *
from models.switch_model import SwitchModel


class SparseConvClusteringSpatial1(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(SparseConvClusteringSpatial1, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries,
                                          learning_rate)
        self.weight_weights = []


    def _compute_output(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        # net = self._placeholder_all_features
        # net = tf.concat((net, self._placeholder_space_features_local), axis=2)

        # TODO: Will cause problems with batch size of 1
        _input = construct_sparse_io_dict(self._placeholder_other_features, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = sparse_conv_make_neighbors(_input, num_neighbors=18, output_all=15, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=15, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=15, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=30, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=10, n_transformed_spatial_features=3, propagrate_ahead=True)

        output = net['spatial_features_global'] * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        self._graph_temp = output

        return output

    def get_variable_scope(self):
        return 'sparse_conv_clustering_spatial1'
