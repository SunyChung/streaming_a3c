import tensorflow as tf 
import os
import scipy.signal
from shutil import copytree

from streaming_test_helper import * 
from streaming_a3c_network import AC_Network
from streaming_env import Streaming_env
from streaming_a3c_single_worker import Worker

from_model_path = '/Users/s/Desktop/streaming_a3c/streaming_a3c_training/a3c_train_10001/model_a3c_10001'
to_model_path = './model_a3c'
if not os.path.exists(to_model_path):
	copytree(from_model_path, to_model_path)

gamma = 0.99
a_size = 10
load_model = True
train = False
minute = 3
repeat = 10
generation_seed = 1
# train = 0
# test_length = 1
# test_mix = 2
wifi_cost = 1
lte_cost = 5

tf.reset_default_graph()

global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
master_network = AC_Network(a_size, 'global', None)
saver = tf.train.Saver(max_to_keep=5)
single_agent = Worker(Streaming_env(minute, repeat, generation_seed, wifi_cost, lte_cost), 0, a_size, repeat, trainer, to_model_path, global_episodes)

env_plotting(minute, repeat, generation_seed)

with tf.Session() as sess:
	if load_model == True:
		print('loading model...')
		ckpt = tf.train.get_checkpoint_state(to_model_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())
	single_agent.work(gamma, sess, saver, train)

actions_graph(minute, repeat, 0)
