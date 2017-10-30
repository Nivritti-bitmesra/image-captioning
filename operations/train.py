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

    def prepare_dataset_iterator(self):
        """
        Prepares iterator operation for operating over dataset in TFRecors
        Returns:
        training_init_op: Intializer for Training Operation
        evaluation_init_op: Intializer for Evaluation Operation
        next_element: Tensor for getting next element in operation
        """
        print("Loading dataset iterators: ")
        train_files = map(lambda current_file: os.path.join(
            self.train_config.train_base_dir, current_file), os.listdir(self.train_config.train_base_dir))
        eval_files = map(lambda current_file: os.path.join(
            self.train_config.eval_base_dir, current_file), os.listdir(self.train_config.eval_base_dir))

        # Training Dataset
        training_dataset = tf.contrib.data.TFRecordDataset(train_files)
        training_dataset = training_dataset.map(self._parse_sequence_example)
        training_dataset = training_dataset.batch(self.train_config.batch_size)

        # Evaluation Dataset
        evaluation_dataset = tf.contrib.data.TFRecordDataset(eval_files)
        evaluation_dataset = evaluation_dataset.map(
            self._parse_sequence_example)
        evaluation_dataset = evaluation_dataset.batch(
            self.train_config.batch_size)

        iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                                           training_dataset.output_shapes)
        next_element = iterator.get_next()

        training_init_op = iterator.make_initializer(training_dataset)
        evaluation_init_op = iterator.make_initializer(evaluation_dataset)

        return training_init_op, evaluation_init_op, next_element

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

    def train_step(self, i, sess, training_op, training_init_op, evaluation_init_op, next_element):
        """"
        Run single step through training process
        """
        start = timeit.default_timer()
        sess.run(training_init_op)
        net_loss = 0
        loop_length = self.train_config.num_examples_per_epoch_train / \
            self.train_config.batch_size

        for k in tqdm(range(loop_length)):
            try:
                image_features, input_sequence, target_sequence, input_mask = sess.run(
                    next_element)
                feed_dict = {
                    "image_features:0": image_features,
                    "input_sequence:0": input_sequence,
                    "target_sequence:0": target_sequence,
                    "input_mask:0": input_mask
                }
                _, merged_summary, total_loss, global_step = sess.run(
                    [training_op, self.model.merged_summary, self.model.total_loss, self.model.global_step], feed_dict=feed_dict)
                #print global_step
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
        training_init_op, evaluation_init_op, next_element = self.prepare_dataset_iterator()
        training_op = self.train_op()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore_model:
                print('Restoring Model: ')
                self.saver.restore(sess,"model_dir/min_loss_model.ckpt")
            self.writer.add_graph(sess.graph)
            for i in range(self.train_config.number_of_epoch):
                print("Running epoch %d" % (i + 1))
                self.train_step(
                    i + 1, sess, training_op, training_init_op, evaluation_init_op, next_element)
