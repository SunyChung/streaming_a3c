import os
from streaming_helper import *
from streaming_env_optimal import Streaming_env_optimal
import shutil

minute = 3
repeat = 10
generation_seed = 1
wifi_cost = 1
lte_cost = 5
result_path = './optimal_results'
if os.path.exists(result_path):
	shutil.rmtree(result_path)
os.mkdir(result_path)

opt_env = Streaming_env_optimal(minute, repeat, generation_seed, wifi_cost, lte_cost, result_path)

episode_count = 0
episodes_qoes = []
episodes_costs = []
episodes_rewards = []
episodes_rebuffering_counts = []

while episode_count < repeat:
	episode_action = []
	episode_qoe = []
	episode_cost = []
	episode_reward = []
	episode_rebuffering_history = []
	d = False
	t = 0
	opt_env.env_reset(episode_count)
	while d == False:
		opt_env.action_list_reset()
		a = opt_env.next_optimal_action(episode_count, t)
		q, c, r, b, d = opt_env.evaluate(episode_count, t)
		episode_action.append(a)
		episode_qoe.append(q)
		episode_cost.append(c)
		episode_reward.append(r)
		episode_rebuffering_history.append(b)
		t += 1

	np.savetxt(os.path.join(result_path, 'test_' + str(episode_count) + '_optimal_actions.txt'), episode_action, fmt='%1d')
	np.savetxt(os.path.join(result_path, 'test_' + str(episode_count) + '_optimal_qoe.txt'), episode_qoe, fmt='%5.3f')
	np.savetxt(os.path.join(result_path, 'test_' + str(episode_count) + '_optimal_cost.txt'), episode_cost, fmt='%5.3f')
	np.savetxt(os.path.join(result_path, 'test_' + str(episode_count) + '_optimal_reward.txt'), episode_reward, fmt='%5.3f')
	np.savetxt(os.path.join(result_path, 'test_' + str(episode_count) + '_optimal_rebuffering.txt'), episode_rebuffering_history, fmt='%5.3f')
	episode_count += 1

	episodes_qoes.append((episode_qoe[-1]))
	episodes_costs.append(episode_cost[-1])
	episodes_rewards.append(np.sum(episode_reward))
	episodes_rebuffering_counts.append(np.sum(1 for x in episode_rebuffering_history if x > 0))

np.savetxt(os.path.join(result_path, 'total_qoes.txt'), episodes_qoes, fmt='%5.3f')
np.savetxt(os.path.join(result_path, 'total_costs.txt'), episodes_costs, fmt='%5.3f')
np.savetxt(os.path.join(result_path, 'total_rewards.txt'), episodes_rewards, fmt='%5.3f')
np.savetxt(os.path.join(result_path, 'total_rebuffering_count.txt'), episodes_rebuffering_counts, fmt='%2d')

result_plotting(generation_seed, minute, repeat)
