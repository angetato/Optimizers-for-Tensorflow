
import os
import os.path
import shutil
import tensorflow as tf
import time
import numpy as np
import random as rn
import datetime
from tensorflow.examples.tutorials.mnist import input_data
from tf_utils.Adam import AdamOptimizer
from tf_utils.NAdam import NAdamOptimizer
from tf_utils.AAdam import AAdamOptimizer
from tf_utils.AAdam01 import AAdamOptimizer01
from tf_utils.AAdam02 import AAdamOptimizer02


LOGDIR = "tensorboard/mnist_LR_8/"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



def mnist_model(learning_rate, opt, hparam):
  tf.reset_default_graph()
  sess = tf.Session()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))	
	
  logits = tf.matmul(x, W) + b

  with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="loss")
    tf.summary.scalar("loss", loss)

  with tf.name_scope("train"):
    if opt == "Adam" : 
        train_step = AdamOptimizer(learning_rate).minimize(loss)
    elif opt == "AAdam01" : 
        train_step = AAdamOptimizer01(learning_rate).minimize(loss)
    elif opt == "AAdam02" : 
        train_step = AAdamOptimizer02(learning_rate).minimize(loss)
    elif opt == "AAdam" : 
        train_step = AAdamOptimizer(learning_rate).minimize(loss)
    else : 
        train_step = NAdamOptimizer(learning_rate).minimize(loss)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()


  '''embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
  assignment = embedding.assign(embedding_input)'''
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOGDIR + hparam)
  writer.add_graph(sess.graph)
	
  writerDev = tf.summary.FileWriter(LOGDIR + hparam+"test")
  writerDev.add_graph(sess.graph)

  '''config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.sprite.image_path = SPRITES
  embedding_config.metadata_path = LABELS
  # Specify the width and height of a single thumbnail.
  embedding_config.sprite.single_image_dim.extend([28, 28])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)'''

  for i in range(20001):
    batch = mnist.train.next_batch(100)
    if i % 250 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)
    if i % 1000 == 0:
      s_test = sess.run(summ, feed_dict={x: mnist.test.images, y: mnist.test.labels})
      writerDev.add_summary(s_test, i)
      #saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, optimizer):
  return "lr_%.0E,%s" % (learning_rate,optimizer)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-2,1E-3]:

    for opt in ["AAdam01","AAdam02", "AAdam", "Adam","NAdam"]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
        hparam = make_hparam_string(learning_rate,opt)
        print('Starting run for %s' % hparam)

        # Actually run with the new settings
        mnist_model(learning_rate, opt, hparam)
  print('Done training!')
  print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
  print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')

if __name__ == '__main__':
  main()
