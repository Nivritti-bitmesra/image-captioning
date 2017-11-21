"""
Configuration Objects for training operation
"""

class ModelConfig(object):
    """ Model Configurations for Image Captioning System"""

    def __init__(self):

        # No of words in the dataset
        self.vocab_size = 1004

        # Image Dimensions
        self.image_height = 224
        self.image_width = 224

        # Random Intializer scale
        self.initializer_scale = 0.08

        # Dimension of Word Embeddings
        self.embedding_size = 200

        # Number of Hidden units in the LSTM
        self.num_lstm_hidden_units = 512

        # Dropout probablity
        self.lstm_dropout_keep_prob = 0.7

        self.image_feature_dimension = 512

        self.fixed_padded_length = 16

class TrainingConfig(object):
    """ Hyperparameter for training"""

    def __init__(self):

        ## Paths to appropriate directories for dataset
        self.train_base_dir = '/media/srivatsa/982ED6FB2ED6D17C/caption_data/train'
        self.eval_base_dir = '/media/srivatsa/982ED6FB2ED6D17C/caption_data/eval'

        ## Paths to appropriate directories for log
        self.log_dir_train = 'log_dir/train'
        self.log_dir_eval = 'log_dir/eval'

        ## Path for saving the dataset
        self.model_path = 'model_dir/min_loss_model.ckpt'

        ## Image Features Dimension
        self.image_features_dimension = 512

        ## No of Epoch
        self.number_of_epoch = 2000

        ## Batch Size
        self.batch_size = 64

        self.num_epochs_per_decay = 8.0

        ## Training dataset size
        self.num_examples_per_epoch_train = 2400*3

        ## Evaluation dataset size
        self.num_examples_per_epoch_eval = 2400

        ## Starting Learning Rate
        self.initial_learning_rate = 0.005

        ## Factor by which learning rate is decayed
        self.learning_rate_decay_factor = 0.5

        self.number_of_epoch_per_decay = 8.0
        self.clip_gradient = 5.0
