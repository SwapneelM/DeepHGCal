from experiment.classification_model_test_result import ClassificationModelTestResult
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler
from models.sparse_conv_cluster_spatial_1 import SparseConvClusteringSpatial1
# from models.sparse_conv_cluster_spatial_2_min_loss import SparseConvClusteringSpatialMinLoss
from models.binning_cluster_alpha_min_loss import BinningClusteringMinLoss
from models.sparse_conv_cluster_spatial_2_min_loss2 import SparseConvClusteringSpatialMinLoss2
from models.sparse_conv_cluster_lstm_based import SparseConvClusterLstmBased
from models.sparse_conv_cluster_seeds_truth_1rand_alpha import SparseConvClusteringSeedsTruthPlusOneRandomAlpha
from models.sparse_conv_cluster_truth_seeds_alpha import SparseConvClusteringSeedsTruthAlpha
from models.sparse_conv_cluster_make_neighbors_new import SparseConvClusteringMakeNeighborsNew
from models.sparse_conv_cluster_bare_baseline import SparseConvClusteringBareBaselineAlpha
from models.sparse_conv_cluster_truth_seeds_beta import SparseConvClusteringSeedsTruthBeta
from models.binning_cluster_beta import BinningClusteringBeta
from models.binning_cluster_gamma import BinningClusteringGamma
import tensorflow as tf
from models.sparse_conv_constant_neighbors_alpha import SparseConvConstantNeighborsAlpha
from models.binning_seed_finder_alpha import BinningSeedFinderAlpha
from models.clustering_dgcnn import DynamicGraphCnnAlpha
from models.binning_cluster_delta import BinningClusteringDelta
from models.binning_cluster_epsilon import BinningClusteringEpsilon
from models.binning_cluster_zeta import BinningClusteringZeta
from models.binning_cluster_eta import BinningClusteringEta


class ModelBuilder:
    def __init__(self, config):
        self.config = config

        # Add check for one_hot_labels - duplicated in TNTuplesClusteringTrainer
        # For God only knows what reason...
        # What was the point of reading in the config in the trainer if you re-read it here...
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
        except KeyError:
            print("KeyError in ModelBuilder")
            model = self.decrypt_model(model_type)
        return model

    def decrypt_model(self, name):
        if name == 'truth_seeds_log_loss':
            model = SparseConvClusteringSeedsTruthBeta(*self.arguments_tuple)
            model.set_input_energy_log(False)
            model.set_loss_energy_function(lambda x: tf.log(x+1))
            return model
        elif name == "truth_seeds_log_energy_log_loss":
            model = SparseConvClusteringSeedsTruthBeta(*self.arguments_tuple)
            model.set_input_energy_log(True)
            model.set_loss_energy_function(lambda x: tf.log(x+1))
            return model
        elif name == "truth_seeds_log_energy":
            model = SparseConvClusteringSeedsTruthBeta(*self.arguments_tuple)
            model.set_input_energy_log(True)
            return model
        elif name == "truth_seeds_seed_talk_off":
            model = SparseConvClusteringSeedsTruthBeta(*self.arguments_tuple)
            model.set_input_energy_log(False)
            model.set_seed_talk(False)
            return model
        elif name == "truth_seeds_normal_energy":
            model = SparseConvClusteringSeedsTruthBeta(*self.arguments_tuple)
            model.set_loss_energy_function(tf.identity)
            return model
        elif name == "truth_seeds_min_loss":
            model = SparseConvClusteringSeedsTruthBeta(*self.arguments_tuple)
            model.set_min_loss_mode(True)
            return model
        elif name == 'BinningSeedFinderAlpha_L2':
            model = BinningSeedFinderAlpha(*self.arguments_tuple)
            model.set_seed_target_l2(True)
            return model
        else:
            raise RuntimeError("Can't find model")
