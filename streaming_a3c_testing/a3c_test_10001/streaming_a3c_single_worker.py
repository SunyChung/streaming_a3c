import os
import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim 

from streaming_test_helper import * 
from streaming_a3c_network import AC_Network

class Worker():
	def __init__(self, env, name, a_size, repeat, trainer, model_path, global_episodes):
		self.result_path = './test_results'
		self.env = env
		self.name = "worker_" + str(name)
		self.number = name
		self.repeat = repeat
		self.trainer = trainer
		self.model_path = model_path
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_qoes = []
		self.episode_costs = []
		self.episode_rewards = []
		self.episode_mean_values = []
		self.episode_rebuffering_count = []
		self.summary_writer = tf.summary.FileWriter("a3c_test_" + str(self.number))
		self.local_AC = AC_Network(a_size, self.name, trainer)
		self.update_local_ops = update_target_graph('global', self.name)
		if not os.path.exists(self.result_path):
			os.makedirs(self.result_path)

	def train(self, rollout, sess, gamma, bootstrap_value):		
	# from the work(): episode_buffer.append([state, a, r, state1, d, v])
		rollout = np.array(rollout)
		states = rollout[:, 0]
		actions = rollout[:, 1]
		rewards = rollout[:, 2]
		next_states = rollout[:, 3]
		values = rollout[:, 5]
		
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages, gamma)

		rnn_state = self.local_AC.state_init
		feed_dict = {self.local_AC.target_v: discounted_rewards,
					self.local_AC.state: np.vstack(states),
					self.local_AC.actions: actions,
					self.local_AC.advantages: advantages,
					self.local_AC.state_in[0]: rnn_state[0],
					self.local_AC.state_in[1]: rnn_state[1]
					}
		v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
												self.local_AC.policy_loss,
												self.local_AC.entropy,
												self.local_AC.grad_norms,
												self.local_AC.var_norms,
												self.local_AC.apply_grads],
									feed_dict=feed_dict)
		return v_l/len(rollout), p_l/len(rollout), e_l/len(rollout), g_n, v_n
		
	def work(self, gamma, sess, saver, train):
		episode_count = sess.run(self.global_episodes)
		nth_repeat = 0
		
		print("starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():

			while nth_repeat < self.repeat:
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_action = []
				episode_qoe = []
				episode_cost = []
				episode_reward = []
				episode_values = []
				episode_rebuffering = []
				episode_step_count = 0
				d = False
				t = 0
				
				s = self.env.reset(nth_repeat)
				state = np.hstack(s)				
				rnn_state = self.local_AC.state_init
				while d == False:
					a_dist, v, rnn_state_new = sess.run([self.local_AC.policy, 
															self.local_AC.value, 
															self.local_AC.state_out],
													feed_dict={self.local_AC.state: [state], 
																self.local_AC.state_in[0]: rnn_state[0], 
																self.local_AC.state_in[1]: rnn_state[1]})
					if t == 0:
						a = 0
					else:
						a = np.random.choice(a_dist[0], p=a_dist[0])
						a = np.argmax(a_dist == a)
					
					rnn_state = rnn_state_new
					s1 = self.env.get_next_state(a, nth_repeat, t)
					state1 = np.hstack(s1)
					q, c, r, d = self.env.evaluate(a, nth_repeat, t)
					
					episode_buffer.append([state, a, r, state1, d, v])
					episode_action.append(a)
					episode_qoe.append(q)
					episode_cost.append(c)
					episode_reward.append(r)
					episode_values.append(v)
					episode_rebuffering.append(s1[2])
					state = state1
					t +=1
					
				self.episode_qoes.append(episode_qoe[-1])
				self.episode_costs.append(episode_cost[-1])
				self.episode_rewards.append(np.sum(episode_reward))
				self.episode_mean_values.append(np.mean(episode_values))
				self.episode_rebuffering_count.append(sum(1 for x in episode_rebuffering if x > 0))

				np.savetxt(os.path.join(self.result_path, \
					'A3C_test_' + str(nth_repeat)  + '_actions.txt'), episode_action, fmt='%1d')
				np.savetxt(os.path.join(self.result_path, \
					'A3C_test_' + str(nth_repeat)  + '_qoes.txt'), episode_qoe, fmt='%5.3f')
				np.savetxt(os.path.join(self.result_path, \
					'A3C_test_' + str(nth_repeat)  + '_costs.txt'), episode_cost, fmt='%5.3f')
				np.savetxt(os.path.join(self.result_path, \
					'A3C_test_' + str(nth_repeat)  + '_rewards.txt'), episode_reward, fmt='%5.3f')
				np.savetxt(os.path.join(self.result_path, \
					'A3C_test_' + str(nth_repeat)  + '_rebuffering_time.txt'), episode_rebuffering, fmt='%5.3f')

				nth_repeat += 1
			
			np.savetxt(os.path.join(self.result_path, \
				'A3C_test_total_qoes.txt'), self.episode_qoes, fmt='%5.3f')
			np.savetxt(os.path.join(self.result_path, \
				'A3C_test_total_costs.txt'), self.episode_costs, fmt='%5.3f')
			np.savetxt(os.path.join(self.result_path, \
				'A3C_test_total_rewards.txt'), self.episode_rewards, fmt='%5.3f')
			np.savetxt(os.path.join(self.result_path, \
				'A3C_test_total_rebuffering_count.txt'), self.episode_rebuffering_count, fmt='%2d')