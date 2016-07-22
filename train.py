import config
import tensorflow as tf
import numpy as np
import os
import preprocessing as pre
import model
import datetime


def train_step(cnn, x_batch, y_batch):
  """
  A single training step
  """
  feed_dict = {
    cnn.input_x: x_batch,
    cnn.input_y: y_batch,
    cnn.dropout_keep_prob: config.dropout_keep_prob
  }
  _, step, summaries, loss, accuracy = cnn.sess.run(
      [cnn.train_op, cnn.global_step, cnn.train_summary_op, cnn.loss, cnn.accuracy],
      feed_dict)

  cnn.train_summary_writer.add_summary(summaries, step)


def val_step(cnn, x_batch, y_batch, writer=None):
  """
  Evaluates model on a val set
  """
  feed_dict = {
    cnn.input_x: x_batch,
    cnn.input_y: y_batch,
    cnn.dropout_keep_prob: 1.0
  }
  loss_avg_op = cnn.val_loss_average.average(cnn.loss)
  step, summaries, loss, loss_avg, accuracy = cnn.sess.run(
      [cnn.global_step, cnn.val_summary_op, cnn.loss, loss_avg_op, cnn.accuracy],
      feed_dict)

  time_str = datetime.datetime.now().isoformat()
  print("{}: step {}, loss {:g}, moving avg {:g}, acc {:g}".format(time_str, step, loss, loss_avg, accuracy))
  if writer:
    writer.add_summary(summaries, step)

  return loss_avg, accuracy


def train():
  print("Loading data...")
  x, y, alphabet = pre.load_data()
  # Randomly shuffle data
  shuffle_indices = np.random.permutation(np.arange(len(y)))
  x_shuffled = x[shuffle_indices]
  y_shuffled = y[shuffle_indices]
  x_train, x_val = x_shuffled[:-1000], x_shuffled[-1000:]
  y_train, y_val = y_shuffled[:-1000], y_shuffled[-1000:]
  print ("Alphabet Size: %d" % len(alphabet))
  print ("Train/Dev split: %d/%d" % (len(y_train), len(y_val)))

  sess = tf.Session()

  sess = tf.Session()
  with sess.as_default():
    cnn = model.SentenceCNN(
        sequence_length=x_train.shape[1],
        num_classes=2,
        alphabet_size=len(alphabet),
        sess=sess
    )
    cnn.inference()
    cnn.train()
    cnn.summary()

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    # Initialize all variables
    sess.run(tf.initialize_all_variables())

    prec_eval_loss_avg = np.inf

    strips = 0
    prec_acc = [0] * config.successive_strips
    # Generate batches
    batches = pre.batch_iter(list(zip(x_train, y_train)), config.batch_size, config.num_epochs)
    # Training loop. For each batch...
    for batch in batches:
      x_batch, y_batch = zip(*batch)
      train_step(cnn, x_batch, y_batch)
      current_step = tf.train.global_step(sess, cnn.global_step)

      if current_step % config.evaluate_every == 0:
        print("\nEvaluation:")
        eval_loss_avg, acc = val_step(cnn, x_val, y_val, writer=cnn.val_summary_writer)

        if eval_loss_avg > prec_eval_loss_avg:  # ) < config.epsilon_val_mov_avg:
          prec_acc[strips] = acc
          strips += 1
        else:
          prec_acc = [0] * config.successive_strips
          strips = 0
        if strips == config.successive_strips:
          print("\nEARLY STOPPING with loss {:g}!!\n".format(max(prec_acc)))
          return
        prec_eval_loss_avg = eval_loss_avg

      if current_step % config.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to %s\n" % path)


def main(_):
  if not config.resume:
    if tf.gfile.Exists(config.out_dir):
      checkpoints_dir = os.path.join(config.out_dir, 'checkpoints')
      summaries_dir = os.path.join(config.out_dir, 'summaries')
      if tf.gfile.Exists(checkpoints_dir):
        tf.gfile.DeleteRecursively(checkpoints_dir)
      tf.gfile.MakeDirs(checkpoints_dir)
      if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
      tf.gfile.MakeDirs(summaries_dir)
    else:
      tf.gfile.MakeDirs(config.out_dir)
    train()
  else:
    print "resume not"


if __name__ == '__main__':
  tf.app.run()