import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os

def optimal_best(minutes, repeat):
	data_path='./optimal_vs_train_10001'

	for i in range(repeat):
		x = np.arange(0, minutes*60, 4)		
		optimal = genfromtxt(os.path.join(data_path, 'test_' + str(i) + '_optimal_actions.txt'))
		best = genfromtxt(os.path.join(data_path, 'A3C_test_' + str(i) + '_actions.txt'))

		plt.figure(figsize=[20, 10])
		plt.plot(x, optimal, '-', label='optimal')
		plt.plot(x, best, '-.', label='A3C best (10000_trained)', linewidth=2)
		plt.xticks(np.arange(0, minutes*60+4, 4))
		plt.yticks(np.arange(10), ('OFF(0.3Mbps)', 'OFF(0.75Mbps)', 'OFF(1.2Mbps)', \
			'OFF(1.85Mbps)', 'OFF(2.85Mbps)', 'ON(0.3Mbps)', 'ON(0.75Mbps)', \
			'ON(1.2Mbps)', 'ON(1.85Mbps)', 'ON(2.85Mbps)'), rotation=45)
		plt.xlabel('timesteps (1 episode = ' + str(minutes*60) + ' seconds)')
		plt.ylabel('actions taken')
		plt.legend(loc='best')
		plt.title('action comparison for test ' + str(i) + ': optimal vs. A3C agent')
		plt.savefig(os.path.join(data_path, 'test_' + str(i) + '_action_comparison.png'))
		plt.close()
	return

optimal_best(3, 10)