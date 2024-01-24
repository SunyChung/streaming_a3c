import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os

def comparison_graph(minute, repeat):
	test_1 = './train_1001'
	test_2 = './train_5001'
	test_3 = './train_10001'
	test_4 = './train_50001'
	#test_5 = './train_100001'
	optimal = './optimal'

	x = genfromtxt(os.path.join(test_1, 'meta-A3C_test_total_rewards.txt'))
	y = genfromtxt(os.path.join(test_2, 'meta-A3C_test_total_rewards.txt'))
	z = genfromtxt(os.path.join(test_3, 'meta-A3C_test_total_rewards.txt'))
	k = genfromtxt(os.path.join(test_4, 'meta-A3C_test_total_rewards.txt'))
	#l = genfromtxt(os.path.join(test_5, 'meta-A3C_test_total_rewards.txt'))
	o = genfromtxt(os.path.join(optimal, 'optimal_total_rewards.txt'))

	plt.figure(figsize=[20, 10])
	plt.plot(x, '-', label='trained_1001_times', linewidth=2)
	plt.plot(y, '-.', label='trained_5001_times', linewidth=2)
	plt.plot(z, ':', label='trained_10001_times', linewidth=3)
	plt.plot(k, '--', label='trained_50001_times', linewidth=2)
	#plt.plot(l, '-.', label='trained_100001_times')
	plt.plot(o, '-', label='optimal', linewidth=2)

	plt.legend(loc='lower left')
	plt.xticks(np.arange(0, repeat,  1))
	plt.xlabel('test trials: ' + str(minute) + '-minute ' + str(repeat) + ' tests')
	plt.ylabel('total reward per test')
	plt.grid(True)
	plt.title('meta-A3C test: reward comparison between (1001/ 5001/ 10001/ 50001)-times trained models vs optimal value')
	plt.savefig('meta-A3C_test_reward_graph.png')
	plt.close()
	return

comparison_graph(3, 10)