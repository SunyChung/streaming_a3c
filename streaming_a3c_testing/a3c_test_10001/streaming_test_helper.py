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

def actions_graph(minutes, repeat, worker_num):
	env_path = './bandwidth_txt'
	result_path = './test_results'
	plotting_path = './test_plotting'
	if not os.path.exists(plotting_path):
			os.makedirs(plotting_path)

	for i in range(repeat):
		z = genfromtxt(os.path.join(result_path, 'A3C_test_' + str(i)  + '_actions.txt'))
		m = genfromtxt(os.path.join(env_path, 'test_' + str(i) + '_wifi_bandwidth.txt'))
		n = genfromtxt(os.path.join(env_path, 'test_' + str(i) + '_lte_bandwidth.txt'))
		x = np.arange(0, minutes*60, 4)

		plt.figure(i, figsize=[20, 10])
		plt.plot(x, z * 5, linewidth=3)
		plt.plot(m)
		plt.plot(n)
		plt.xticks(np.arange(0, minutes*60+4, 4), rotation=45)
		plt.yticks(np.arange(0, 50, 5), ('0 Mbps | OFF(0.3Mbps)', '5 Mbps | OFF(0.75Mbps)', '10 Mbps | OFF(1.2Mbps)', \
			'15 Mbps | OFF(1.85Mbps)', '20 Mbps | OFF(2.85Mbps)', '25 Mbps | ON(0.3Mbps)', '30 Mbps | ON(0.75Mbps)', \
			'35 Mbps | ON(1.2Mbps)', '40 Mbps | ON(1.85Mbps)', '45 Mbps | ON(2.85Mbps)'))
		plt.xlabel('action steps (1 episode == ' + str(minutes*60) +' seconds) with WiFi & LTE bandwidth (Mbps)')
		plt.title('A3C agent action for test_' + str(i) + ' with wifi & LTE bandwidth')
		plt.grid(True)
		plt.savefig(os.path.join(plotting_path, 'A3C_test_' + str(i) + '_action_with_bandwdith.png'))
		plt.close()

		"""
		plt.figure(i * 13, figsize=[15, 10])
		z = genfromtxt(os.path.join(result_path, 'A3C_test_' + str(i)  + '_costs.txt'))
		x = np.arange(0, minutes*60, 4)
		plt.plot(x, z)
		plt.xticks(np.arange(0, minutes*60+4, 4), rotation=45)
		plt.xlabel('testing steps (1 episode == ' + str(minutes*60) +' seconds)')
		plt.ylabel('costs')
		plt.title('A3C test episode_' + str(i) + '_cost graph')
		plt.grid(True)
		plt.savefig(os.path.join(plotting_path, 'A3C_test_' + str(i)  + '_costs.png'))
		plt.close()

		plt.figure(i * 17, figsize=[15, 10])
		z = genfromtxt(os.path.join(result_path, 'A3C_test_' + str(i)  + '_qoes.txt'))
		x = np.arange(0, minutes*60, 4)
		plt.plot(x, z)
		plt.xticks(np.arange(0, minutes*60+4, 4), rotation=45)
		plt.xlabel('testing steps (1 episode == ' + str(minutes*60) +' seconds)')
		plt.ylabel('qoes')
		plt.title('A3C test episode_' + str(i) + '_QoE graph')
		plt.grid(True)
		plt.savefig(os.path.join(plotting_path, 'A3C_test_' + str(i)  + '_qoes.png'))
		plt.close()	

		plt.figure(i * 19, figsize=[15, 10])
		z = genfromtxt(os.path.join(result_path, 'A3C_test_' + str(i)  + '_rewards.txt'))
		x = np.arange(0, minutes*60, 4)
		plt.plot(x, z)
		plt.xticks(np.arange(0, minutes*60+4, 4), rotation=45)
		plt.xlabel('testing steps (1 episode == ' + str(minutes*60) +' seconds)')
		plt.ylabel('rewards')
		plt.title('A3C test episode_' + str(i) + '_reward graph')
		plt.grid(True)
		plt.savefig(os.path.join(plotting_path, 'A3C_test_' + str(i)  + '_rewards.png'))
		plt.close()	
		"""
	return None

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
		self.bandwidth_graph(self.minutes, self.repeat)
		return 

	def bandwidth_graph(self, minutes, repeat):
		for i in range(repeat):
			x = genfromtxt(os.path.join(self.env_path, 'test_' + str(i) + '_wifi_bandwidth.txt'))
			plt.figure(1 + i, figsize=[20, 10])
			plt.plot(x)
			plt.xticks(np.arange(0, minutes*60+4, 4))       
			plt.xlabel('testing time(' + str(minutes*60) +' seconds)')
			plt.ylabel('WiFi bandwidth (Mbps)')
			plt.title('WiFi bandwidth for test_' + str(i))
			plt.grid(True)
			plt.savefig(os.path.join(self.plotting_path, 'test_' + str(i) + '_wifi_bandwidth.png'))

			y = genfromtxt(os.path.join(self.env_path, 'test_' + str(i) + '_lte_bandwidth.txt'))
			plt.plot(y)
			plt.xticks(np.arange(0, minutes*60+4, 4))       
			plt.xlabel('testing time(' + str(minutes*60) +' seconds)')
			plt.ylabel('WiFi & LTE bandwidth (Mbps)')
			plt.title('WiFi & LTE bandwidth for test_' + str(i))
			plt.grid(True)
			plt.savefig(os.path.join(self.plotting_path, 'test_' + str(i) + '_wifi_&_lte_bandwidth.png'))
			plt.close()
		return None