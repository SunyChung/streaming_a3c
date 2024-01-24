import numpy as np
import itertools as it
import matplotlib.pyplot as plt 
from numpy import genfromtxt
import os

class Streaming_env():
	def __init__(self, minute, repeat, generation_seed, wifi_cost, lte_cost):
		self.minutes = minute
		self.seconds = self.minutes * 60
		self.steps = self.minutes * 15
		self.repeat = repeat
		self.wifi_throughput = np.zeros((self.repeat, self.steps+1, 4))
		self.lte_throughput = np.zeros((self.repeat, self.steps+1, 4))
		self.wifi_unit_cost = wifi_cost
		self.lte_unit_cost = lte_cost
		self.set_seed = generation_seed
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
			self.wifi_throughput[i] = np.reshape(wifi_bandwidth, (self.steps+1, 4))

			lte_bandwidth = (np.random.normal(35, 5, self.seconds+4))
			self.lte_throughput[i] = np.reshape(lte_bandwidth, (self.steps+1, 4))
		return self.wifi_throughput, self.lte_throughput

	def reset(self, nth_repeat):
		bitrate_0 = 0.3 * np.ones(4)
		bandwidth_0 = self.wifi_throughput[nth_repeat][0]
		self.rebuffering_time = np.zeros((self.steps+1))
		self.rebuffering_time[0] = np.sum(bitrate_0/bandwidth_0)
		self.bitrate_history = np.zeros((self.steps+1))
		self.bitrate_history[0] = 0.3
		self.bitrate_dif = np.zeros((self.steps+1))
		self.num_chunks_left = (self.steps-1)
		self.episode_qoe = np.zeros((self.steps+1))
		self.episode_qoe[0] = 0.3 - self.rebuffering_time[0]
		self.episode_cost = np.zeros((self.steps+1))
		self.episode_cost[0] = np.sum(self.wifi_unit_cost * np.asarray(bandwidth_0))
		return self.wifi_throughput[nth_repeat][0], self.lte_throughput[nth_repeat][0], self.rebuffering_time[0], self.num_chunks_left, self.bitrate_history[0]
	
	def get_next_state(self, action, nth_repeat, timestep):
		next_wifi = self.wifi_throughput[nth_repeat][timestep+1]
		next_lte = self.lte_throughput[nth_repeat][timestep+1]

		if action == 0:
			alpha = 0
			self.bitrate_history[timestep+1] = 0.3
		elif action == 1:
			alpha = 0
			self.bitrate_history[timestep+1] = 0.75
		elif action == 2:
			alpha = 0
			self.bitrate_history[timestep+1] = 1.2
		elif action == 3:
			alpha = 0
			self.bitrate_history[timestep+1] = 1.85
		elif action == 4:
			alpha = 0
			self.bitrate_history[timestep+1] = 2.85
		elif action == 5:
			alpha = 1
			self.bitrate_history[timestep+1] = 0.3
		elif action == 6:
			alpha = 1
			self.bitrate_history[timestep+1] = 0.75
		elif action == 7:
			alpha = 1
			self.bitrate_history[timestep+1] = 1.2
		elif action == 8:
			alpha = 1
			self.bitrate_history[timestep+1] = 1.85
		elif action == 9:
			alpha = 1
			self.bitrate_history[timestep+1] = 2.85		
		
		next_bitrate = self.bitrate_history[timestep+1]
		self.bitrate_dif[timestep+1] = abs(next_bitrate - self.bitrate_history[timestep])

		chunk_bitrate = next_bitrate * np.ones(4)
		bandwidth_sum = np.asarray(next_wifi) + (alpha * np.asarray(next_lte))	
		download_time = np.sum(chunk_bitrate/bandwidth_sum)
		buffering = max(0, (download_time - 4))
		self.rebuffering_time[timestep+1] = buffering
		
		self.num_chunks_left = 15 - (timestep+1)
		return next_wifi, next_lte, self.rebuffering_time[timestep+1], self.num_chunks_left, next_bitrate
			
	def evaluate(self, action, nth_repeat, timestep):
		next_wifi = self.wifi_throughput[nth_repeat][timestep+1]
		next_lte = self.lte_throughput[nth_repeat][timestep+1]

		if action >= 5:
			alpha = 1
		else:
			alpha = 0

		np.seterr(all='print')
		qoe = self.bitrate_history[timestep+1] - self.rebuffering_time[timestep+1] - self.bitrate_dif[timestep+1]
		self.episode_qoe[timestep+1] = qoe
		cost = np.sum((self.wifi_unit_cost * np.asarray(next_wifi)) + (alpha * self.lte_unit_cost * np.asarray(next_lte)))
		self.episode_cost[timestep+1] = cost
		reward = np.divide(np.sum(self.episode_qoe) + qoe, np.sum(self.episode_cost) + cost, dtype=float)
		
		if (timestep) == (self.steps-1):
			done = True
		else:
			done = False
		return qoe, cost, reward, done