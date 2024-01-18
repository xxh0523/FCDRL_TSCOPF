import numpy as np
import torch
import pathlib
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.use('Agg')


def split_equal(total: int, num: int):
	quotient = int(total / num)
	remainder = total % num
	start = list()
	end = list()
	for i in range(num):
		if i < remainder:
			start.append(i*(quotient+1))
			end.append((i+1)*(quotient+1))
		else:
			start.append(i*quotient+remainder)
			end.append((i+1)*quotient+remainder)
	return start, end


class Logger:
    def __init__(self, logfile_path: pathlib.Path, train_log_name=None, test_log_name=None):
        self.training_logfile = None if train_log_name is None else open(logfile_path / train_log_name, 'w')
        self.testing_logfile = None if test_log_name is None else open(logfile_path / test_log_name, 'w')
    
    def __del__(self):
        if self.training_logfile is not None: self.training_logfile.close()
        if self.testing_logfile is not None: self.testing_logfile.close()
    
    def training_log(self, *strs):
        string = ' '.join(strs)
        self.training_logfile.write(string + '\n')
        tqdm.write(string)
    
    def testing_log(self, *strs):
        string = ' '.join(strs)
        self.testing_logfile.write(string + '\n')
        tqdm.write(string) 


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.p = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		ind[0] = np.argmax(self.reward)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class Picture_Drawer:
	def __init__(self):
		pass

	@staticmethod
	def draw_sopf_train_test(
		data_path: pathlib.Path, 
		n_processor: int, 
		total_step: int, 
		interval: int, 
		size=12,
		):
		font1 = {'size': size}
		mpl.rcParams['xtick.labelsize'] = size
		mpl.rcParams['ytick.labelsize'] = size
		
		training_data = np.load(data_path, allow_pickle=True)
		train = training_data['train'].reshape(-1,n_processor)
		eval = training_data['eval']
		x = np.arange(1, total_step+1)
		plt.xlim((0, total_step+1))
		plt.ylim((np.min(eval), 1000))
		plt.xlabel('Training Step', fontdict=font1)
		plt.ylabel('Average Reward', fontdict=font1)
		plt.tick_params(labelsize=size)
		avg_train_reward = np.sum(train, axis=1) / n_processor
		plt.scatter(x, avg_train_reward,
		            s = 5,
		            label="training", 
		            c = 'b',
		            )
		# for i in range(n_processor):
		# 	plt.scatter(x, train[:, i],
		#             	s = 5,
		#             	label=f"training_processor{i}", 
		#             	c = 'b',
		#             	)
		x = np.arange(0, total_step+1, interval)
		avg_test_reward = np.average(eval[:x.shape[0]], axis=1)
		plt.plot(x, avg_test_reward,
		         label="evaluation", 
		         color = 'r',
		         )
		plt.legend(loc='best')
		plt.savefig(data_path.parent / 'training.jpg', dpi=300, bbox_inches='tight', format='jpg')
		plt.close()

