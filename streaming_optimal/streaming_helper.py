import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import os

def bandwidth_graph(env_path, plotting_path, minutes, repeat):
	for i in range(repeat):
		x = genfromtxt(os.path.join(env_path, 'test_' + str(i) + '_wifi_bandwidth.txt'))
		plt.figure(1 + i, figsize=[20, 10])
		plt.plot(x, label='wifi')
		plt.xticks(np.arange(0, minutes*60+4, 4))       
		plt.xlabel('testing time (seconds)')
		plt.ylabel('WiFi bandwidth (Mbps)')
		plt.title('WiFi bandwidth for testing')
		plt.grid(True)
		plt.savefig(os.path.join(plotting_path, 'test_' + str(i) + '_wifi_bandwidth.png'))

		y = genfromtxt(os.path.join(env_path, 'test_' + str(i) + '_lte_bandwidth.txt'))
		plt.plot(y, label='LTE')
		plt.xticks(np.arange(0, minutes*60+4, 4))       
		plt.xlabel('testing time (seconds)')
		plt.ylabel('WiFi & LTE bandwidth (Mbps)')
		plt.title('WiFi & LTE bandwidth for testing')
		plt.legend(loc='best')
		plt.grid(True)
		plt.savefig(os.path.join(plotting_path, 'test_' + str(i) + '_wifi_&_lte_bandwidth.png'))
		plt.close()
	return None

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def result_plotting(generation_seed, minutes, repeat):
	env_path = './bandwidth_txt'
	result_path = './optimal_results'
	result_plotting = './optimal_results_plotting'
	if not os.path.exists(result_plotting):
		os.makedirs(result_plotting)

	for i in range(repeat):
		xset = np.arange(0, minutes*60, 4)
		x = genfromtxt(os.path.join(result_path, 'test_' + str(i) + '_optimal_actions.txt'))
		y = np.arange(0, minutes*60 + 4, 1)
		m = genfromtxt(os.path.join(env_path, 'test_' + str(i) + '_wifi_bandwidth.txt'))
		n = genfromtxt(os.path.join(env_path, 'test_' + str(i) + '_lte_bandwidth.txt'))

		fig = plt.figure(figsize=[20, 10])
		ax1 = fig.add_subplot(1, 1, 1)
		ax1.plot(xset, 5 * x + 5, linewidth=4, color='green')
		ax1.axhline(y=25, color='black', linestyle='--')
		ax1.set_xlabel('action steps (1 episode == ' + str(minutes*60) +' seconds) with wifi & LTE bandwidth (Mbps)')
		ax1.set_xticks(np.arange(0, minutes*60 + 4, 4))       		
		ax1.grid(True)
		ytick = ['', 'OFF(0.3Mbps)', 'OFF(0.75Mbps)', 'OFF(1.2Mbps)', 'OFF(1.85Mbps)', 'OFF(2.85Mbps)',\
			 'ON(0.3Mbps)', 'ON(0.75Mbps)', 'ON(1.2Mbps)', 'ON(1.85Mbps)', 'ON(2.85Mbps)', '']
		plt.yticks(np.arange(0, 55, 5), ytick)

		ax2 = ax1.twinx()
		ax2.plot(y, m, y, n)
		ax2.set_yticks(np.arange(0, 55, 5))
		plt.title('optimal action for test_' + str(i) + 'with wifi & LTE bandwidth')
		plt.savefig(os.path.join(result_plotting, 'test_' + str(i) + '_optimal_action_with_bandwidth.png'))
		plt.close()
	return 
