import tensorflow as tf 
import numpy as np 
import random
import scipy.misc
import itertools as it
import matplotlib.pyplot as plt 
from numpy import genfromtxt
import os

def update_target_graph(from_scope, to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

	op_holder = []
	for from_var, to_var in zip(from_vars, to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std/ np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

class env_plotting():
	def __init__(self, minute, repeat, generation_seed):
		self.minutes = minute
		self.seconds = self.minutes * 60
		self.steps = self.minutes * 15
		self.repeat = repeat
		self.wifi_throughput = np.zeros((self.repeat, self.steps+1, 4))
		self.lte_throughput = np.zeros((self.repeat, self.steps+1, 4))
		self.set_seed = generation_seed
		self.env_path = './bandwidth_txt'
		self.plotting_path = './bandwidth_plotting'
		if not os.path.exists(self.env_path):
			os.makedirs(self.env_path)
		if not os.path.exists(self.plotting_path):
			os.makedirs(self.plotting_path)
		self.throughput_generate()

	def index_generator(self):
		indices = []
		for index in it.permutations(range(self.seconds+4), 3):
			if sum(index)==(self.seconds+4):
				indices.append(index)
		return indices 

	def throughput_generate(self):
		np.random.seed(self.set_seed)
		index_list = self.index_generator()
		indices = np.random.choice(len(index_list), self.repeat)
		
		for i in range(self.repeat):
			index = indices[i]
			wifi_bandwidth_low = (np.random.normal(1, 0.1, index_list[index][0]))
			wifi_bandwidth_med = (np.random.normal(7, 0.1, index_list[index][1]))
			wifi_bandwidth_app = (np.random.normal(15, 0.1, index_list[index][2]))
			
			list = [wifi_bandwidth_low, wifi_bandwidth_med, wifi_bandwidth_app]
			list_order = [e for e in (it.permutations(list))]
			list_choice = np.random.choice(len(list_order), 1).item()
			wifi_bandwidth = np.append(list_order[list_choice][0], list_order[list_choice][1])
			wifi_bandwidth = np.append(wifi_bandwidth, list_order[list_choice][2])
			np.savetxt(os.path.join(self.env_path, 'test_' + str(i) + '_wifi_bandwidth.txt'), wifi_bandwidth, fmt='%3.3f')
			self.wifi_throughput[i] = np.reshape(wifi_bandwidth, (self.steps+1, 4))

			lte_bandwidth = (np.random.normal(35, 5, self.seconds+4))
			np.savetxt(os.path.join(self.env_path, 'test_' + str(i) + '_lte_bandwidth.txt'), lte_bandwidth, fmt='%3.3f')
			self.lte_throughput[i] = np.reshape(lte_bandwidth, (self.steps+1, 4))
		return None

	def bandwidth_graph(self, minutes, repeat):
		for i in range(self.repeat):
			x = genfromtxt(os.path.join(self.env_path, 'train_' + str(i) + '_wifi_bandwidth.txt'))
			plt.figure(1, figsize=[20, 10])
			plt.plot(x)
			plt.xticks(np.arange(0, minutes*60+4, 1))
			plt.xlabel('training time (seconds)')
			plt.ylabel('WiFi bandwidth (Mbps)')
			plt.title('WiFi bandwidth for train_' + str(i))
			plt.grid(True)
			plt.savefig(os.path.join(self.plotting_path, 'train_' + str(i) + '_wifi.png'))

			y = genfromtxt(os.path.join(self.env_path, 'train_' + str(i) + '_lte_bandwidth.txt'))
			plt.plot(y)
			plt.xticks(np.arange(0, minutes*60+4, 1))
			plt.xlabel('training time (seconds)')
			plt.ylabel('WiFi & LTE bandwidth (Mbps)')
			plt.title('WiFi & LTE bandwidth for train_' + str(i))
			plt.grid(True)
			plt.savefig(os.path.join(self.plotting_path, 'train_' + str(i) + '_wifi+lte.png'))
			plt.close()
		return None