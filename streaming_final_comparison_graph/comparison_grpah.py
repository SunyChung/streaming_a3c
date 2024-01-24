import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os

def final_graph(minute, repeat):
	optimal_path ='./optimal'
	a3c_path = './optimal_vs_a3c_train_10001'
	meta_a3c_path = './optimal_vs_meta_a3c_train_10001'
	result_path = './final_graph'
	if not os.path.exists(result_path):
		os.makedirs(result_path)

	for i in range(repeat):
		optimal_qoe = genfromtxt(os.path.join(optimal_path, 'test_' + str(i) + '_optimal_qoe.txt'))
		optimal_cost = genfromtxt(os.path.join(optimal_path, 'test_' + str(i) + '_optimal_cost.txt'))
		optimal_reward = genfromtxt(os.path.join(optimal_path, 'test_' + str(i) + '_optimal_reward.txt'))

		a3c_qoe = genfromtxt(os.path.join(a3c_path, 'A3C_test_' + str(i) + '_qoes.txt'))
		a3c_cost = genfromtxt(os.path.join(a3c_path, 'A3C_test_' + str(i) + '_costs.txt'))
		a3c_reward = genfromtxt(os.path.join(a3c_path, 'A3C_test_' + str(i) + '_rewards.txt'))

		meta_a3c_qoe = genfromtxt(os.path.join(meta_a3c_path, 'meta-A3C_test_' + str(i) + '_qoes.txt'))
		meta_a3c_cost = genfromtxt(os.path.join(meta_a3c_path, 'meta-A3C_test_' + str(i) + '_costs.txt'))
		meta_a3c_reward = genfromtxt(os.path.join(meta_a3c_path, 'meta-A3C_test_' + str(i) + '_rewards.txt'))

		plt.figure(i, figsize=[20, 10])
		plt.plot(optimal_qoe, 'r', label='optimal_qoe')
		plt.plot(a3c_qoe, '-.y', label='a3c_qoe', linewidth=4)
		plt.plot(meta_a3c_qoe, '--g', label='meta_a3c_qoe')
		plt.legend(loc='best')
		plt.xticks(np.arange(0, minute*60+4, 4))
		plt.xlabel('testing time (1 episode == ' + str(minute*60) +' seconds)')
		plt.ylabel('qoe values')
		plt.title('test_' + str(i) + '_qoe_comparison: optimal vs. A3C vs. meta-A3C')
		plt.grid(True)
		plt.savefig(os.path.join(result_path, 'test_' + str(i) + '_qoe_comparison.png'))
		plt.close()

		plt.figure(i + 1, figsize=[20, 10])
		plt.plot(optimal_cost, 'r', label='optimal_cost')
		plt.plot(a3c_cost, '-.y', label='a3c_cost', linewidth=4)
		plt.plot(meta_a3c_cost, '--g', label='meta_a3c_cost')
		plt.legend(loc='best')
		plt.xticks(np.arange(0, minute*60+4, 4))
		plt.xlabel('testing time (1 episode == ' + str(minute*60) +' seconds)')
		plt.ylabel('costs')
		plt.title('test_' + str(i) + '_cost_comparison: optimal vs. A3C vs. meta-A3C')
		plt.grid(True)
		plt.savefig(os.path.join(result_path, 'test_' + str(i) + '_cost_comparison.png'))
		plt.close()

		plt.figure(i + 2, figsize=[20, 10])
		plt.plot(optimal_reward, 'r', label='optimal_reward')
		plt.plot(a3c_reward, '-.y', label='a3c_reward', linewidth=4)
		plt.plot(meta_a3c_reward, '--g', label='meta_a3c_reward')
		plt.legend(loc='best')
		plt.xticks(np.arange(0, minute*60+4, 4))
		plt.xlabel('testing time (1 episode == ' + str(minute*60) +' seconds)')
		plt.ylabel('reward values')
		plt.title('test_' + str(i) + '_reward_comparison: optimal vs. A3C vs. meta-A3C')
		plt.grid(True)
		plt.savefig(os.path.join(result_path, 'test_' + str(i) + '_reward_comparison.png'))
		plt.close()
	return 

#final_graph(3, 10)