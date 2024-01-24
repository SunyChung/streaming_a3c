import os
import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim 

from streaming_train_helper import * 
from streaming_a3c_network import AC_Network

class Worker():
	def __init__(self, env, name, a_size, repeat, trainer, model_path, global_episodes):
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
		self.summary_writer = tf.summary.FileWriter("a3c_train_" + str(self.number))
		self.local_AC = AC_Network(a_size, self.name, trainer)
		self.update_local_ops = update_target_graph('global', self.name)

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
		
	def work(self, gamma, sess, coord, saver, train):
		episode_count = sess.run(self.global_episodes)
		
		print("starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():
		
			while not coord.should_stop() and episode_count != self.repeat:
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_action = []
				episode_qoe = []
				episode_cost = []
				episode_reward = []
				episode_values = []
				episode_step_count = 0
				d = False
				t = 0
				
				s = self.env.reset(episode_count)
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
					s1 = self.env.get_next_state(a, episode_count, t)
					state1 = np.hstack(s1)
					q, c, r, d = self.env.evaluate(a, episode_count, t)
					
					episode_buffer.append([state, a, r, state1, d, v])
					episode_action.append(a)
					episode_qoe.append(q)
					episode_cost.append(c)
					episode_reward.append(r)
					episode_values.append(v)
					state = state1
					t +=1
					
				self.episode_qoes.append(episode_qoe[-1])
				self.episode_costs.append(episode_cost[-1])
				self.episode_rewards.append(np.sum(episode_reward))
				self.episode_mean_values.append(np.mean(episode_values))

				if len(episode_buffer) != 0 and train == True:
					v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

				if episode_count != 0:
					if episode_count %10 == 0 and self.name =='worker_0' and train == True:
						saver.save(sess, self.model_path + '/model_' + str(episode_count) + '.cptk')
						print("saved model")
					
					mean_qoe = np.mean(self.episode_qoes[-5:])
					mean_cost = np.mean(self.episode_costs[-5:])
					mean_reward = np.mean(self.episode_rewards[-5:])
					mean_value = np.mean(self.episode_mean_values[-5:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Qoe', simple_value=float(mean_qoe))
					summary.value.add(tag='Perf/Cost', simple_value=float(mean_cost))
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					if train  == True:
						summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
						summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
						summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
						summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
						summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)
					self.summary_writer.flush()

				if self.name == 'worker_0':
					sess.run(self.increment)
				episode_count += 1
			