import threading
import multiprocessing
import numpy as np 
import tensorflow as tf 
import scipy.signal
import os
from time import sleep
from time import time 

from streaming_train_helper import * 
from streaming_meta_a3c_network import AC_Network
from streaming_env import Streaming_env
from streaming_meta_a3c_worker import Worker

repeat = 10001
gamma = 0.99
a_size = 10
model_path = './model_meta_a3c_' + str(repeat)
load_model = False
train = True
minute = 1
generation_seed = 0
# train = 0
# test_length = 1
# test_mix = 2
wifi_cost = 1
lte_cost = 5

tf.reset_default_graph()

if not os.path.exists(model_path):
	os.makedirs(model_path)

with tf.device("/cpu:0"):
	global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
	trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
	master_network = AC_Network(a_size, 'global', None)
	num_workers = multiprocessing.cpu_count()
	workers = []
	for i in range(num_workers):
		workers.append(Worker(Streaming_env(minute, repeat, generation_seed, wifi_cost, lte_cost), i, a_size, repeat, trainer, model_path, global_episodes))
	saver = tf.train.Saver(max_to_keep=5)

env_plotting(minute, repeat, generation_seed)

with tf.Session() as sess:
	coord = tf.train.Coordinator()

	if load_model == True:
		print('loading model...')
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())

	worker_threads = []
	for worker in workers:
		worker_work = lambda: worker.work(gamma, sess, coord, saver, train)
		thread = threading.Thread(target=(worker_work))
		thread.start()
		sleep(0.1)
		worker_threads.append(thread)
	coord.join(worker_threads)
