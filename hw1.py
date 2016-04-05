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

	def forward(self, ob_list, time):
		"""Use forward algorithm to evaluate probability of a given observation.

		Parameters
		----------
		ob_list : array-like, Observation list.
		time : integer, Assign which time of observation list.

		Returns
		-------
		p : float, Probability of given observation.
		"""
		if time > len(ob_list):
			raise IndexError("Time cannot be more than length of observation list.")

		ob_list = [self.ob_list.index(i) for i in ob_list]	# Transform observation to integer type
		# Calculate probability of first observation for every state
		prob_list = [self.init_prob[i] * self.ob_prob[i][ob_list[0]] for i in range(self.state_num)]

