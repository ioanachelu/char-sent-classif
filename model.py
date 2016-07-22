import tensorflow as tf
import numpy as np
import config
import time
import os

class SentenceCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, alphabet_size, sess):
      self.sequence_length = sequence_length
      self.num_classes = num_classes
      self.alphabet_size = alphabet_size
      self.sess = sess

      # Placeholders for input, output and dropout
      self.input_x = tf.placeholder(tf.int32, [None, alphabet_size, sequence_length], name="input_x")
      self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Keeping track of l2 regularization loss (optional)
      self.l2_loss = tf.constant(0.0)


    def inference(self):
      # network weights
      convolution_weights_1 = tf.Variable(tf.truncated_normal([self.alphabet_size, 7, 256], stddev=0.05))
      convolution_bias_1 = tf.Variable(tf.constant(0.05, shape=[256]))

      hidden_convolutional_layer_1 = tf.nn.relu(
          tf.nn.conv2d(self.input_x, convolution_weights_1, strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_1)

      hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 256, 3, 1],
                                                  strides=[1, 3, 3, 1], padding="SAME")

      convolution_weights_2 = tf.Variable(tf.truncated_normal([256, 7, 256], stddev=0.05))
      convolution_bias_2 = tf.Variable(tf.constant(0.05, shape=[256]))

      hidden_convolutional_layer_2 = tf.nn.relu(
          tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 1, 1, 1],
                       padding="SAME") + convolution_bias_2)

      hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 256, 3, 1],
                                                  strides=[1, 3, 3, 1], padding="SAME")

      convolution_weights_3 = tf.Variable(tf.truncated_normal([256, 3, 256], stddev=0.05))
      convolution_bias_3 = tf.Variable(tf.constant(0.05, shape=[256]))

      hidden_convolutional_layer_3 = tf.nn.relu(
          tf.nn.conv2d(hidden_max_pooling_layer_2, convolution_weights_3,
                       strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_3)


      convolution_weights_4 = tf.Variable(tf.truncated_normal([256, 3, 256], stddev=0.05))
      convolution_bias_4 = tf.Variable(tf.constant(0.05, shape=[256]))

      hidden_convolutional_layer_4 = tf.nn.relu(
          tf.nn.conv2d(hidden_convolutional_layer_3, convolution_weights_4,
                       strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_4)


      convolution_weights_5 = tf.Variable(tf.truncated_normal([256, 3, 256], stddev=0.05))
      convolution_bias_5 = tf.Variable(tf.constant(0.05, shape=[256]))

      hidden_convolutional_layer_5 = tf.nn.relu(
          tf.nn.conv2d(hidden_convolutional_layer_4, convolution_weights_5,
                       strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_5)


      convolution_weights_6 = tf.Variable(tf.truncated_normal([256, 3, 256], stddev=0.05))
      convolution_bias_6 = tf.Variable(tf.constant(0.05, shape=[256]))

      hidden_convolutional_layer_6 = tf.nn.relu(
          tf.nn.conv2d(hidden_convolutional_layer_5, convolution_weights_6,
                       strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_6)

      hidden_max_pooling_layer_6 = tf.nn.max_pool(hidden_convolutional_layer_6, ksize=[1, 256, 3, 1],
                                                  strides=[1, 3, 3, 1], padding="SAME")

      hidden_max_pooling_layer_6_shape = hidden_max_pooling_layer_6.get_shape()[1] * \
                                         hidden_max_pooling_layer_6.get_shape()[2] * \
                                         hidden_max_pooling_layer_6.get_shape()[3]
      hidden_max_pooling_layer_6_shape = hidden_max_pooling_layer_6_shape.value

      hidden_convolutional_layer_6_flat = tf.reshape(hidden_max_pooling_layer_6, [-1, hidden_max_pooling_layer_6_shape])

      feed_forward_weights_7 = tf.Variable(tf.truncated_normal([hidden_max_pooling_layer_6_shape, 1024], stddev=0.05))
      feed_forward_bias_7 = tf.Variable(tf.constant(0.05, shape=[1024]))

      feed_forward_layer_7 = tf.nn.relu(
          tf.matmul(hidden_convolutional_layer_6_flat, feed_forward_weights_7) + feed_forward_bias_7)

      # Add dropout
      h_drop_1 = tf.nn.dropout(feed_forward_layer_7, self.dropout_keep_prob)

      feed_forward_weights_8 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.05))
      feed_forward_bias_8 = tf.Variable(tf.constant(0.05, shape=[1024]))

      feed_forward_layer_8 = tf.nn.relu(
          tf.matmul(h_drop_1, feed_forward_weights_8) + feed_forward_bias_8)

      # Add dropout
      h_drop_2 = tf.nn.dropout(feed_forward_layer_8, self.dropout_keep_prob)

      feed_forward_weights_9 = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.05))
      feed_forward_bias_9 = tf.Variable(tf.constant(0.05, shape=[2]))

      self.output_layer = tf.matmul(h_drop_2, feed_forward_weights_9) + feed_forward_bias_9



    def loss(self):

        # self.l2_loss += tf.nn.l2_loss(W)
        # self.l2_loss += tf.nn.l2_loss(b)
        # self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
      self.predictions = tf.argmax(self.output_layer, 1, name="predictions")

      # CalculateMean cross-entropy loss
      with tf.name_scope("cross-entropy"):
        losses = tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.input_y)
        self.loss = tf.reduce_mean(losses)
                    # + config.l2_reg_lambda * self.l2_loss
        self.val_loss_average = tf.train.ExponentialMovingAverage(0.9999, name='val_loss_mov_avg')
        self.val_loss_average_op = self.val_loss_average.apply([self.loss])

      # Accuracy
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def train(self):
      self.global_step = tf.Variable(0, name="global_step", trainable=False)

      with tf.control_dependencies([self.val_loss_average_op]):
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


    def summary(self):
      # Keep track of gradient values and sparsity (optional)
      grad_summaries = []
      for grad, var in self.grads_and_vars:
        if grad is not None:
          grad_hist_summary = tf.histogram_summary(var.op.name + '/gradients/hist', grad)
          sparsity_summary = tf.scalar_summary(var.op.name + '/gradients/sparsity', tf.nn.zero_fraction(grad))
          grad_summaries.append(grad_hist_summary)
          grad_summaries.append(sparsity_summary)

      grad_summaries_merged = tf.merge_summary(grad_summaries)

      # Output directory for models and summaries
      timestamp = str(int(time.time()))
      print("Writing to %s\n" % config.out_dir)

      # Summaries for loss and accuracy
      loss_summary = tf.scalar_summary("loss", self.loss)
      acc_summary = tf.scalar_summary("accuracy", self.accuracy)

      # Train Summaries
      self.train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
      train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
      self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, self.sess.graph_def)

      # Dev summaries
      self.val_summary_op = tf.merge_summary([loss_summary, acc_summary])
      val_summary_dir = os.path.join(config.out_dir, "summaries", "val")
      self.val_summary_writer = tf.train.SummaryWriter(val_summary_dir, self.sess.graph_def)


