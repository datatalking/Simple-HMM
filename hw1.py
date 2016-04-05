class HMM:
	"""docstring for HMM"""
	def __init__(self, state_num, observation_list):
		self.state_num = state_num
		self.init_prob = [0 for i in range(state_num)]									# Initial probability for choosing first state
		self.state = [i for i in range(state_num)]
		self.state_prob = [[0 for j in range(state_num)] for i in range(state_num)]		# Every state's transition probability
		self.ob_list = observation_list.copy()
		self.ob_num = len(observation_list)
		self.ob_prob = [[0 for j in range(self.ob_num)] for i in range(self.state_num)]	# Every state's observation probability


		