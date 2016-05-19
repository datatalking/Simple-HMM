from math import log, exp


class HMM:
    """Simple implement for Hidden Markov Model"""
    def __init__(self, state_num, observation_list,
                 initial_probability=None, transition_probability=None, observation_probability=None):
        self.state_num = state_num
        # Initial probability for choosing first state
        self._init_prob = [0 for i in range(state_num)] if not initial_probability else initial_probability
        self._state = [i for i in range(state_num)]
        # Every state's transition probability
        self._state_prob = [[(1/self.state_num) for j in range(state_num)] for i in range(state_num)] \
            if not transition_probability else transition_probability
        self._ob_list = observation_list
        self._ob_num = len(observation_list)
        # Every state's observation probability
        self._ob_prob = [[1/self._ob_num for j in range(self._ob_num)] for i in range(self.state_num)] \
            if not observation_probability else observation_probability
        # Translate probability to log
        self._init_prob = [log(p) for p in self._init_prob]
        self._state_prob = [[log(p) for p in state] for state in self._state_prob]
        self._ob_prob = [[log(p) for p in state] for state in self._ob_prob]

    def forward(self, ob_list, time):
        """Use forward algorithm to evaluate probability of a given observation.

        Parameters
        ----------
        ob_list : array-like, Observation list.
        time : integer, Assign which time of observation list.

        Returns
        -------
        p : float, Probability of given observation.
        prob_list : array, Forward probability in every time stamp.
        """
        if time > len(ob_list):
            raise IndexError("Time cannot be more than length of observation list.")

        ob_list = self._get_ob_index(ob_list)  # Transform observation to index
        # Calculate probability of first observation for every state
        forward_prob = [self._initial_ob_prob(ob_list[0])]

        for t in range(1, time):
            forward_prob.append([])
            for j in range(self.state_num):
                # Calculate probability that previous state probability transit to present state probability
                p = self._log_sum([forward_prob[t-1][i] + self._state_prob[i][j] for i in range(self.state_num)])
                # Calculate probability that present state to present observation
                forward_prob[t].append(p + self._ob_prob[j][ob_list[t]])

        return exp(self._log_sum(forward_prob[time-1])), forward_prob

    def backward(self, ob_list, time):
        """Use backward algorithm to evaluate probability of a given observation.

        Parameters
        ----------
        ob_list : array-like, Observation list.
        time : integer, Assign which time of observation list.

        Returns
        -------
        p : float, Probability of given observation.
        prob_list : array, Backward probability in every time stamp.
        """
        if time > len(ob_list):
            raise IndexError("Time cannot be more than length of observation list.")

        ob_list = self._get_ob_index(ob_list)  # Transform observation to index
        # Initialize the probability
        backward_prob = [[log(1) for i in range(self.state_num)] for t in range(time)]

        for t in range(time-2, -1, -1):
            for i in range(self.state_num):
                # Calculate probability that following state probability back to present state probability.
                p = self._log_sum([backward_prob[t+1][j] + self._state_prob[i][j] + self._ob_prob[j][ob_list[t+1]]
                                   for j in range(self.state_num)])
                backward_prob[t][i] = p
        # Return the probability from last time to first time (need to multiply the probability from time0 to time1)
        return exp(self._log_sum([self._init_prob[i] + self._ob_prob[i][ob_list[0]] + backward_prob[0][i]
                                  for i in range(self.state_num)])), backward_prob

    def decode(self, ob_list, time):
        """Use viterbi algorithm to find the state sequence for a given observation list.

        Parameters
        ----------
        ob_list : array-like, Observation list.
        time : integer, Assign which time of observation list.

        Returns
        -------
        state_seq : array, The best state sequence for given observation list
        """
        if time > len(ob_list):
            raise IndexError("Time cannot be more than length of observation list.")

        ob_list = self._get_ob_index(ob_list)   # Transform observation to index
        # Calculate probability of first observation for every state
        max_prob = self._initial_ob_prob(ob_list[0])
        pre_prob = max_prob[:]
        path = [[i] for i in range(self.state_num)]
        for t in range(1, time):
            new_path = [[] for i in range(self.state_num)]
            for j in range(self.state_num):
                # Find maximum probability and the most possible previous state to transit to present state
                p, state = max([(pre_prob[i] + self._state_prob[i][j], i) for i in range(self.state_num)])
                # Calculate probability that present state to present observation
                max_prob[j] = p + self._ob_prob[j][ob_list[t]]
                # Choose the most possible path to present state
                new_path[j] = path[state] + [j]
            # Record for changed probability
            pre_prob = max_prob[:]
            # Record for new path
            path = new_path
        # Find the last state
        (prob, state) = max([(max_prob[i], i) for i in range(self.state_num)])
        
        return path[state]

    def train(self, data_sets):
        """Use EM algorithm to train models.

        Parameters
        ----------
        data_sets : array-like, A array of observation list.

        Returns
        -------
        """
        size = len(data_sets)
        # The probability of every path which pass through state i
        all_state_prob = []   # gamma
        # The probability of every path which pass by route from state i to state j
        all_stateset_prob = []  # xi
        # initial_prob = [0 for i in range(self.state_num)]
        # state_prob = [[0 for j in range(self.state_num)] for i in range(self.state_num)]

        for data in data_sets:
            time = len(data)
            # The probability of every path which pass through state i
            state_prob = [-1e10 for i in range(self.state_num)]
            state_prob = [state_prob for t in range(time)]    # gamma
            # The probability of every path which pass by route from state i to state j
            state_set_prob = [[-1e10 for j in range(self.state_num)] for i in range(self.state_num)]
            state_set_prob = [state_set_prob for t in range(time)]    # xi

            _, forward_prob = self.forward(data, time)
            _, backward_prob = self.backward(data, time)
            data = self._get_ob_index(data)

            for t, ob in enumerate(data):
                # p += α[t][i] * β[t][i]
                p = self._log_sum([forward_prob[t][i] + backward_prob[t][i] for i in range(self.state_num)])
                # γ[t][i] = α[t][i] * β[t][i] / p
                for i in range(self.state_num):
                    state_prob[t][i] = forward_prob[t][i] + backward_prob[t][i] - p

                if t < time-1:
                    # p += α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j]
                    p = self._log_sum([forward_prob[t][i] + self._state_prob[i][j] +
                                       self._ob_prob[j][data[t+1]] + backward_prob[t+1][j]
                                       for i in range(self.state_num) for j in range(self.state_num)])
                    # ξ[t][i][j] = α[t][i] * a[i][j] * b[j][o[t+1]] * β[t+1][j] / p;
                    for i in range(self.state_num):
                        for j in range(self.state_num):
                            state_set_prob[t][i][j] = forward_prob[t][i] + self._state_prob[i][j] + \
                                                     self._ob_prob[j][data[t+1]] + backward_prob[t+1][j] - p
            # Update initial probability
            """
            self._init_prob = [state_prob[0][i] for i in range(self.state_num)]

            # Update state transition probability
            for i in range(self.state_num):
                p2 = self._log_sum([state_prob[t][i] for t in range(time-1)])
                for j in range(self.state_num):
                    p1 = self._log_sum([state_set_prob[t][i][j] for t in range(time-1)])
                    self._state_prob[i][j] = p1 - p2

            # Update observation probability
            for i in range(self.state_num):
                p = [-1e10 for o in range(self._ob_num)]
                p2 = self._log_sum([state_prob[t][i] for t in range(time)])
                for t in range(time):
                    p[data[t]] = self._log_sum([p[data[t]], state_prob[t][i]])

                for j in range(self._ob_num):
                    self._ob_prob[i][j] = p[j] - p2
    """

            all_state_prob.append(state_prob)
            all_stateset_prob.append(state_set_prob)
        pi = [self._log_sum([all_state_prob[l][0][i] for l in range(size)]) - log(size)
              for i in range(self.state_num)]
        print("pi:", pi)
        a = [[-1e10 for i in range(self.state_num)] for j in range(self.state_num)]
        b = [[-1e10 for o in range(self._ob_num)] for j in range(self.state_num)]
        for i in range(self.state_num):
            p2 = self._log_sum([all_state_prob[l][t][i] for l in range(size) for t in range(len(data_sets[l]) - 1)])
            for j in range(self.state_num):
                p1 = self._log_sum([all_stateset_prob[l][t][i][j]
                                    for l in range(size) for t in range(len(data_sets[l]) - 1)])
                print([all_stateset_prob[l][t][i][j] for l in range(size) for t in range(len(data_sets[l]) - 1)])
                a[i][j] = p1 - p2

        for i in range(self.state_num):
            p = [-1e10 for o in range(self._ob_num)]
            p2 = self._log_sum([all_state_prob[l][t][i] for l in range(size) for t in range(len(data_sets[l]))])
            for l in range(size):
                for t in range(len(data_sets[l])):
                    ob_ind = self._ob_list.index(data_sets[l][t])
                    p[ob_ind] = self._log_sum([p[ob_ind], all_state_prob[l][t][i]])

            for j in range(self._ob_num):
                b[i][j] = p[j] - p2

        self._init_prob = pi
        self._state_prob = a
        self._ob_prob = b
        """
        self._init_prob = [self._log_sum([all_state_prob[l][0][i] - log(size)
                                          for l in range(size)])
                           for i in range(self.state_num)]
        for i in range(self.state_num):
            p2 = -1.0E10
            p = 0
            for x in all_state_prob:
                p+=sum(x[0:-1][i])
            p = [-1.0E10 for i in range(self._ob_num)]
            for s, x in enumerate(all_state_prob):
                for t, y in enumerate(x):
                    ob_ind = self._ob_list.index(data_sets[s][t])
                    p[ob_ind] = self._log_sum((p[ob_ind], y[i]))
                    p2 = self._log_sum((p2, y[i]))

            for j in range(self.state_num):
                p1 = -1.0E10
                for prob_list in all_stateset_prob:
                    for prob in prob_list[:-1]:
                        p1 = self._log_sum((p1, prob[i][j]))
                self._state_prob[i][j] = p1 - p2

            for j in range(self._ob_num):
                self._ob_prob[i][j] = p[j] - p2
        """

    def _get_ob_index(self, observation):
        return [self._ob_list.index(i) for i in observation]  # Transform observation to index

    def _initial_ob_prob(self, ob_index):
        return [self._init_prob[i] + self._ob_prob[i][ob_index] for i in range(self.state_num)]

    @staticmethod
    def _log_sum(sequence):
        """
        :param sequence: array-like, The log number need to be add like log(p+q)
        :return: integer, After calculate log(p+q)
        """
        start = sequence[0]
        for value in sequence[1:]:
            if start < value:
                start, value = value, start
            if start == 0 and value == 0:
                start = log(exp(start) + exp(value))
            else:
                start += log(1 + exp(value - start))
        return start
