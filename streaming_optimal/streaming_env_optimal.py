import os
import numpy as np
import itertools as it
import matplotlib.pyplot as plt 
from numpy import genfromtxt

from streaming_helper import bandwidth_graph

class Streaming_env_optimal():
	def __init__(self, minute, repeat, generation_seed, wifi_cost, lte_cost, result_path):
		self.env_path = './bandwidth_txt'
		self.plotting_path = './bandwidth_plotting'
		if not os.path.exists(self.env_path):
			os.makedirs(self.env_path)
		if not os.path.exists(self.plotting_path):
			os.makedirs(self.plotting_path)

		self.result_path = result_path
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
			np.savetxt(os.path.join(self.env_path, 'test_' + str(i) + '_wifi_bandwidth.txt'), wifi_bandwidth, fmt='%3.3f')
			self.wifi_throughput[i] = np.reshape(wifi_bandwidth, (self.steps+1, 4))

			lte_bandwidth = (np.random.normal(35, 5, self.seconds+4))
			np.savetxt(os.path.join(self.env_path, 'test_' + str(i) + '_lte_bandwidth.txt'), lte_bandwidth, fmt='%3.3f')
			self.lte_throughput[i] = np.reshape(lte_bandwidth, (self.steps+1, 4))
		bandwidth_graph(self.env_path, self.plotting_path, self.minutes, self.repeat)
		return 
	
	def env_reset(self, nth_repeat):
		bitrate_0 = 0.3 * np.ones(4) 
		bandwidth_0 = self.wifi_throughput[nth_repeat][0]
		self.rebuffering_history = np.zeros((self.steps+1))
		self.rebuffering_history[0] = np.sum(bitrate_0/bandwidth_0)
		self.bitrate_history = np.zeros((self.steps+1))
		self.bitrate_history[0] = 0.3
		self.bitrate_dif = np.zeros((self.steps+1))
		self.optimal_qoe = np.zeros((self.steps+1))
		self.optimal_qoe[0] = 0.3 - self.rebuffering_history[0]
		self.optimal_cost = np.zeros((self.steps+1))
		self.optimal_cost[0] = np.sum(self.wifi_unit_cost * np.asarray(bandwidth_0))
		self.optimal_reward = np.zeros((self.steps+1))
		self.optimal_reward[0] = np.divide(self.optimal_qoe[0], self.optimal_cost[0], dtype=float)
		return 

	def action_list_reset(self):
		self.qoe_list = []
		self.cost_list = []
		self.reward_list = []
		self.rebuffering_list = []
		return 
		
	def next_optimal_action(self, nth_repeat, timestep):
		current_bitrate = self.bitrate_history[timestep]
		next_wifi = self.wifi_throughput[nth_repeat][timestep+1]
		next_lte = self.lte_throughput[nth_repeat][timestep+1]
		action_list = [(0, 0.3), (0, 0.75), (0, 1.2), (0, 1.85), (0, 2.85), (1, 0.3), (1, 0.75), (1,  1.2), (1, 1.85), (1, 2.85)]

		for alpha, next_bitrate in action_list:
			chunk_bitrate = next_bitrate * np.ones(4) 
			bandwidth_sum = np.asarray(next_wifi) + (alpha * np.asarray(next_lte))
			download_time = np.sum(chunk_bitrate/bandwidth_sum)
			rebuffering_time = max(0, (download_time - 4))

			qoe =  next_bitrate - rebuffering_time - abs(next_bitrate - self.bitrate_history[timestep]) 
			cost =  np.sum((self.wifi_unit_cost * np.asarray(next_wifi)) + (alpha * self.lte_unit_cost * np.asarray(next_lte)))
			reward = np.divide((np.sum(self.optimal_qoe) + qoe), (np.sum(self.optimal_cost) + cost), dtype=float)

			self.qoe_list.append(qoe)
			self.cost_list.append(cost)
			self.reward_list.append(reward)
			self.rebuffering_list.append(rebuffering_time)
		
		max_reward = max(self.reward_list)
		action_index = self.reward_list.index(max_reward)

		expected_rebuffering = self.rebuffering_list[action_index]
		self.rebuffering_history[timestep+1] = expected_rebuffering

		next_bitrate = action_list[action_index][1]
		self.bitrate_history[timestep+1] = next_bitrate
		self.bitrate_dif[timestep+1] = abs(next_bitrate - self.bitrate_history[timestep])

		qoe_result = self.qoe_list[action_index]
		cost_result = self.cost_list[action_index]
		reward_result = self.reward_list[action_index]

		self.optimal_qoe[timestep+1] = qoe_result
		self.optimal_cost[timestep+1] = cost_result
		self.optimal_reward[timestep+1] = reward_result
		if timestep == 0:
			return 0
		else:
			return action_index
		
	def evaluate(self, nth_repeat, timestep):
		qoe= self.optimal_qoe[timestep]
		cost = self.optimal_cost[timestep]
		reward = self.optimal_reward[timestep]
		rebuffering = self.rebuffering_history[timestep]

		if timestep == (self.steps-1):
			np.savetxt(os.path.join(self.result_path,  'test_'+ str(nth_repeat) + '_bitrate_history.txt'), self.bitrate_history, fmt='%5.3f')
			np.savetxt(os.path.join(self.result_path,  'test_'+ str(nth_repeat) + '_rebuffering_history.txt'), self.rebuffering_history, fmt='%5.3f')
			np.savetxt(os.path.join(self.result_path,  'test_'+ str(nth_repeat) + '_bitrate_dif.txt'), self.bitrate_dif, fmt='%5.3f')
			done = True
		else: 
			done = False

		return qoe, cost, reward, rebuffering, done
			