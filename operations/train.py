import tensorflow as tf
import numpy as np
import os
import timeit
from tqdm import tqdm


class TrainCaptionModel(object):
    def __init__(self, model, train_config):

        self.train_config = train_config
        self.model = model
        self.train_writer = tf.summary.FileWriter(
            self.train_config.log_dir_train)
        self.eval_writer = tf.summary.FileWriter(
            self.train_config.log_dir_eval)

        self.iterator_intializer = self._setup_and_build_model()
        self.training_op = self._train_op()
        self.saver = tf.train.Saver()
        self.min_loss = None
        self.counter_train = 0
        self.counter_eval = 0

    def _get_file_list(self, base_dir):
        """
        Lists all the files inside the base directory
        """
        files = map(lambda current_file: os.path.join(base_dir, current_file),
                    os.listdir(base_dir))
        return files

    def _parse_sequence_example(self, serialized):
        context, sequence = tf.parse_single_sequence_example(serialized, context_features={
            "image/vgg16_features": tf.FixedLenFeature([self.train_config.image_features_dimension], dtype=tf.float32)
        },
            sequence_features={
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

        image_features = context["image/vgg16_features"]
        caption_sequence = sequence["image/caption_ids"]

        input_sequence = caption_sequence[:-1]
        output_sequence = caption_sequence[1:]

        input_mask = tf.to_int32(tf.not_equal(
            input_sequence, 0, name="mask_generate"))

        return image_features, input_sequence, output_sequence, input_mask

    def _train_op(self):
        """
        Set up training operation
        """
        print("Setting up training operations: ")
        variable_learning_rate = self.train_config.initial_learning_rate
        if self.train_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (self.train_config.num_examples_per_epoch_train /
                                     self.train_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              self.train_config.num_epochs_per_decay)

            variable_learning_rate = tf.train.exponential_decay(learning_rate=self.train_config.initial_learning_rate,
                                                                global_step=self.model.global_step,
                                                                decay_steps=decay_steps,
                                                                decay_rate=self.train_config.learning_rate_decay_factor,
                                                                staircase=True)

        optimizer = tf.train.AdamOptimizer(variable_learning_rate)
        grads_and_var = optimizer.compute_gradients(self.model.total_loss)
        clipped_grads = [(tf.clip_by_value(grad, -self.train_config.clip_gradient,
                                           self.train_config.clip_gradient), var) for grad, var in grads_and_var]
        training_op = optimizer.apply_gradients(clipped_grads)

        return training_op

    def _setup_and_build_model(self):
        """
        Add input graph to the overall model and build the entire graph.
        """
        with tf.device('/gpu:0'):
            print('Setting up input: ')
            file_names = tf.placeholder(
                name="file_names", dtype=tf.string, shape=[None])

            dataset = tf.data.TFRecordDataset(file_names)
            dataset = dataset.map(self._parse_sequence_example)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.batch(self.train_config.batch_size)
            iterator = dataset.make_initializable_iterator()

            image_features, input_sequence, target_sequence, input_mask = iterator.get_next()

            parameter_map = {
                "image_features": image_features,
                "input_sequence": input_sequence,
                "target_sequence": target_sequence,
                "input_mask": input_mask
            }
            self.model.feed_input(parameter_map)
            self.model.build()

        return iterator.initializer

    def _train_step(self, i, sess):
        """"
        Run single step through training process
        """
        start = timeit.default_timer()
        net_loss = 0.0

        # Training
        print('Train:')
        loop_length = self.train_config.num_examples_per_epoch_train / \
            self.train_config.batch_size
        feed_dict = {"file_names:0": self._get_file_list(
            self.train_config.train_base_dir)}
        sess.run(self.iterator_intializer, feed_dict=feed_dict)

        for k in tqdm(range(loop_length)):
            try:
                _, total_loss, merged_summary = sess.run([self.training_op, self.model.total_loss,
                                                          self.model.merged_summary])
                net_loss += total_loss
            except tf.errors.OutOfRangeError:
                break
            self.train_writer.add_summary(merged_summary, self.counter_train)
            self.counter_train += 1
        print('Train Loss: %f' % net_loss)

        #self.counter = 0
        net_loss = 0.0
        
        # Evaluation
        print('Eval: ')
        feed_dict = {"file_names:0": self._get_file_list(
            self.train_config.eval_base_dir)}
        sess.run(self.iterator_intializer, feed_dict=feed_dict)
        loop_length = self.train_config.num_examples_per_epoch_eval / \
            self.train_config.batch_size

        for k in tqdm(range(loop_length)):
            try:
                total_loss, merged_summary = sess.run([self.model.total_loss,
                                                       self.model.merged_summary])
                net_loss += total_loss
            except tf.errors.OutOfRangeError:
                break
            self.eval_writer.add_summary(merged_summary, self.counter_eval)
            self.counter_eval += 1
            if self.min_loss > net_loss or self.min_loss is None:
                self.min_loss = net_loss
                save_path = self.saver.save(sess,self.train_config.model_path)

        print("Total loss after %dth iteration is %f. Duration: %ds" %
              (i, net_loss, timeit.default_timer() - start))

    def train_model(self, restore_model=False):
        """
        Function to train the model
        """
        print("Running training operation: ")
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if restore_model:
                print('Restoring Model: ')
                self.saver.restore(sess, "model_dir/min_loss_model.ckpt")
            self.train_writer.add_graph(sess.graph)
            for i in range(self.train_config.number_of_epoch):
                print("Running epoch %d" % (i + 1))
                self._train_step(i + 1, sess)
