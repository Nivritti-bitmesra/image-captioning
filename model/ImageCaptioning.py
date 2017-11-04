import numpy as np
import tensorflow as tf
from tensorflow_vgg.vgg16 import Vgg16


class ImageCaptioning(object):

    def __init__(self, config, mode):
        """
        Intial Setups
        """

        if mode == "inference":
            self.vgg_model = Vgg16()
        self.config = config
        self.intializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale
        )
        self.mode = mode

        # All Image Related Variables

        # A float32 tensor of shape [height, width, channels]
        self.inference_image = None

        # A float32 tensor of shape [batch_size,feature_size]
        self.image_features = None
        # A float32 tensor with shape [batch_size, embedding_size]
        self.image_embeddings = None
        # All Caption Related values
        # A int32 tensor with shape [batch_size, padded_length]
        self.input_sequence = None
        # A int32 tensor with shape [batch_size, padded_length, embedding_size]
        self.input_sequence_embeddings = None
        # A int32 tensor with shape [batch_size, padded_length]
        self.target_sequence = None
        # A int32 tensor with shape [batch_size, padded_length]
        self.input_mask = None


        # All Losses
        # A float32 scalar
        self.total_loss = None
        # A float32 tensor with shape [batch_size*padded_length]
        self.target_cross_entropy_loss = None
        # A float32 tensor with shape [batch_size*padded_length]
        self.target_cross_entropy_loss_weights = None

        # All softmax probablities
        # A float32 tensor of shape [padded_length, vocab_size]
        self.softmax_score = None

        # Global Step Variable
        self.global_step = None

    def feed_input(self, parameter_map):
        """
        param:
        parameter_map: A map containing input data as per the condition.

        Add appropriate parameters as per following condition
        1) In Training mode:
            # self.image_features: VGG16 Features
            # self.input_sequence: Input Sequence of Caption
            # self.target_sequence: Target Sequence of Caption
            # self.input_mask: Mask for Input Sequence of Caption
        2) In Testing mode:
            # self.image_features: VGG16 Features
            # self.input_feed: The input to LSTM cell
            # self.state_feed: The state input to LSTM cell
        3) In Inference mode:
            # self.inference_image: The input image
            # self.input_feed: The input to LSTM cell
            # self.state_feed: The state input to LSTM cell
        """
        print("Feeding inputs in %s mode" % (self.mode))

        try:
            if self.mode == 'train':
                self.image_features = parameter_map['image_features']
                self.input_sequence = parameter_map['input_sequence']
                self.target_sequence = parameter_map['target_sequence']
                self.input_mask = parameter_map['input_mask']
            elif self.mode == 'test':
                self.image_features = parameter_map['image_features']
                self.input_feed = parameter_map['input_feed']
                self.state_feed = parameter_map['state_feed']
            else:
                self.inference_image = parameter_map['inference_image']
                self.input_feed = parameter_map['input_feed']
                self.state_feed = parameter_map['state_feed']
        except KeyError as ex:
            print('The following parameter was not passed: %s' % e.args[0])

    def build_image_features_graph(self, image):
        """
        Function to generate FC7 layer feature of VGG16 architecture
        Args:
        image : Input image (In BGR Mode)
        Returns:
        image_features: Encoded image features
        """
        print("Building image feature graph")
        image_features = None
        with tf.name_scope("VGG16_features"):
            image = tf.div(image, 255.0)
            vgg16.build(image)
            image_features = vgg16.relu7

        return image_features

    def build_image_sequence_transformation_graph(self, image_features):
        """
        Function to transorm image features from feature space to embedding space
        Args:
        image_features
        Returns:
        image_embeddings
        """
        print("Building image sequence transformation graph")
        image_embeddings = None
        with tf.variable_scope("image_embedding_space_transform") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=image_features,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.intializer,
                biases_initializer=self.intializer,
                scope=scope
            )
        return image_embeddings

    def build_caption_sequence_map_graph(self, input_seq):
        """
        Function to map input sequence to vocabulary embeddings space

        Args:
        input_seq
        Returns:
        input_sequence_embeddings
        """
        print("Building caption sequence map graph")
        input_sequence_embeddings = None
        with tf.variable_scope("sequence_embedding"):
            embedding_map = tf.get_variable(
                name="embedding_map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.intializer
            )
            input_sequence_embeddings = tf.nn.embedding_lookup(
                embedding_map, input_seq)
        return input_sequence_embeddings

    def build_model_graph(self, image_embeddings, input_sequence_embeddings, target_sequence=None, input_mask=None):
        """
        Create the Image Caption model.

        Inputs:
          image_embeddings
          input_sequence_embeddings
          target_sequence (Training Only)
          input_mask (Training Only)

        Outputs:
          total_loss
          target_cross_entropy_loss
          target_cross_entropy_loss_weights
        """
        print("Building caption model graph")
        # LSTM Defination
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_hidden_units,
            state_is_tuple=True
        )

        if(self.mode == "train"):
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob
            )

        with tf.variable_scope("lstm", self.intializer) as lstm_scope:
            # Set the intial state of the variable
            # This basically generates initial state of the lstm_cell
            # first, the run one iteration of lstm_cell with image embedding
            # as the input to generate h0 state
            zero_state = lstm_cell.zero_state(
                batch_size=tf.shape(image_embeddings)[0], dtype=tf.float32)
            _, intial_state = lstm_cell(image_embeddings, zero_state)

            # Allow the reuse of the LSTM variables
            lstm_scope.reuse_variables()
            if self.mode == "inference":
                # Run single LSTM cell
                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.sequeeze(input_sequence_embedding, [1]),
                    state=state_tuple
                )
            else:
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=input_sequence_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=intial_state,
                                                    # Transfor output of LSTM cell from lstm output space(Basically Hidden states)
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)

        # [batch_size, padded_length, output_size] -> [batch_size*padded_length, output_size]
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        # to vocabulary space
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.config.vocab_size,
                activation_fn=None,
                weights_initializer=self.intializer,
                scope=logits_scope
            )

        # Compute approprite output
        # 1) Softmax score in case of inference and test mode
        # 2) Total Loss, Average Batch Loss in case of training mode

        if self.mode == "inference" or self.mode == "test":
            with tf.name_scope("LossComputation"):
                softmax_score = tf.nn.softmax(logits, name="softmax")
            return softmax_score
        else:
            with tf.name_scope('LossComputation'):
                targets = tf.reshape(self.target_sequence, [-1])
                weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

                # Loss Computation
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                        logits=logits)
                average_batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                            tf.reduce_sum(weights),
                                            name="average_batch_loss")

                # Adds the regularization
                tf.losses.add_loss(average_batch_loss)
                total_loss = tf.losses.get_total_loss()

                # Add Summaries
                tf.summary.scalar(self.mode + '_total_loss', total_loss)
                tf.summary.scalar(
                    self.mode + '_average_batch_loss', average_batch_loss)

            return total_loss, losses, weights

    def setup_global_step(self):
        print("Setting up global step")
        global_step = tf.Variable(initial_value=0,
                                  name="global_step",
                                  trainable=False,
                                  collections=[
                                      tf.GraphKeys.GLOBAL_STEP,
                                      tf.GraphKeys.GLOBAL_VARIABLES])
        return global_step

    def build(self):
        """
        Build the overall computation graph based on the conditions
        """
        # Build overall graph with following conditions
        # 1) In inference mode add a node for generating Image Features
        # 2) In inference mode and test mode compute only softmax scores
        with tf.device('/cpu:0'):
            print("Building graph for %s mode" % (self.mode))

            if self.mode == "inference":
                self.image_features = self.build_image_features_graph(
                    self.inference_image)

            self.image_embeddings = self.build_image_sequence_transformation_graph(
                self.image_features)
            self.input_sequence_embeddings = self.build_caption_sequence_map_graph(
                self.input_sequence)
            packed_values = self.build_model_graph(self.image_embeddings,
                                                   self.input_sequence_embeddings,
                                                   self.target_sequence,
                                                   self.input_mask)

            if self.mode == "train":
                self.total_loss, self.target_cross_entropy_loss, self.target_cross_entropy_loss_weights = packed_values
                self.global_step = self.setup_global_step()
            else:
                self.softmax_score = packed_values

            self.merged_summary = tf.summary.merge_all()
