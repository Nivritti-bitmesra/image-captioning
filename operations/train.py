import tensorflow as tf
import numpy as np
import os
import timeit
from tqdm import tqdm


class TrainCaptionModel(object):
    def __init__(self, model, train_config):

        self.train_config = train_config
        self.model = model
        self.writer = tf.summary.FileWriter(self.train_config.log_dir)
        self.iterator_intializer = self.setup_and_build_model()
        self.training_op = self.train_op()
        self.saver = tf.train.Saver()
        self.min_loss = None
        self.counter = 0

    def _parse_sequence_example(self, serialized):
        context, sequence = tf.parse_single_sequence_example(serialized, context_features={
            "image/vgg16_features": tf.FixedLenFeature([512], dtype=tf.float32)
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

    def train_op(self):
        """
        Set up training operation
        """
        print("Setting up training operations: ")
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
        training_op = tf.train.AdamOptimizer(
            variable_learning_rate).minimize(self.model.total_loss)

        return training_op

    def setup_and_build_model(self):
        """
        Add input graph to the overall model and build the entire graph.
        """
        print('Setting up input: ')
        file_names = tf.placeholder(
            name="file_names", dtype=tf.string, shape=[None])

        dataset = tf.data.TFRecordDataset(file_names)
        dataset = dataset.map(self._parse_sequence_example)
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

    def train_step(self, i, sess):
        """"
        Run single step through training process
        """
        start = timeit.default_timer()
        net_loss = 0
        loop_length = self.train_config.num_examples_per_epoch_train / \
            self.train_config.batch_size

        train_files = map(lambda current_file: os.path.join(
                               self.train_config.train_base_dir, current_file),
                               os.listdir(self.train_config.train_base_dir))

        feed_dict = {"file_names:0": train_files}
        sess.run(self.iterator_intializer, feed_dict=feed_dict)

        for k in tqdm(range(loop_length)):
            try:
                _, merged_summary, total_loss, global_step = sess.run(
                    [self.training_op,
                     self.model.merged_summary,
                     self.model.total_loss,
                     self.model.global_step])

                if self.min_loss > total_loss or self.min_loss is None:
                    self.min_loss = total_loss
                    save_path = self.saver.save(
                        sess, "model_dir/min_loss_model.ckpt")
                net_loss += total_loss
            except tf.errors.OutOfRangeError:
                break

            self.writer.add_summary(merged_summary, self.counter)
            self.counter += 1

        # sess.run(evaluation_init_op)

        # total_loss = 0
        # count = 0
        # loop_length = self.train_config.num_examples_per_epoch_eval/self.train_config.batch_size
        # for k in tqdm(range(loop_length)):
        #     try:
        #         image_features, input_sequence, target_sequence, input_mask = sess.run(
        #             next_element)
        #         feed_dict = {
        #             "image_features:0": image_features,
        #             "input_sequence:0": input_sequence,
        #             "target_sequence:0": target_sequence,
        #             "input_mask:0": input_mask
        #         }
        #         loss, merged_summary = sess.run(
        #             [self.model.total_loss, self.model.merged_summary], feed_dict=feed_dict)
        #         total_loss += loss
        #     except tf.errors.OutOfRangeError:
        #         break
        #     #self.writer.add_summary(merged_summary,k)

        print("Total loss after %dth iteration is %f. Duration: %ds" %
              (i, net_loss, timeit.default_timer() - start))

    def train_model(self, restore_model=False):
        """
        Function to train the model
        """
        print("Running training operation: ")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore_model:
                print('Restoring Model: ')
                self.saver.restore(sess, "model_dir/min_loss_model.ckpt")
            self.writer.add_graph(sess.graph)
            for i in range(self.train_config.number_of_epoch):
                print("Running epoch %d" % (i + 1))
                self.train_step(i + 1, sess)
