class HMM:
    """Simple implement for Hidden Markov Model"""
    def __init__(self, state_num, observation_list, initial_probability=None, transition_probability=None, observation_probability=None):
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
        self._forward_prob = []

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

        ob_list = self._get_ob_index(ob_list)  # Transform observation to index
        # Calculate probability of first observation for every state
        forward_prob = [self._initial_ob_prob(ob_list[0])]

        for t in range(1, time):
            forward_prob.append([])
            for j in range(self.state_num):
                # Calculate probability that previous state probability transit to present state probability
                p = sum([forward_prob[t-1][i] * self._state_prob[i][j] for i in range(self.state_num)])
                # Calculate probability that present state to present observation
                forward_prob[t].append(p * self._ob_prob[j][ob_list[t]])
        # Record in class attribute
        self._forward_prob = forward_prob

        return sum(forward_prob[time-1])

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
                p, state = max([(pre_prob[i] * self._state_prob[i][j], i) for i in range(self.state_num)])
                # Calculate probability that present state to present observation
                max_prob[j] = p * self._ob_prob[j][ob_list[t]]
                # Choose the most possible path to present state
                new_path[j] = path[state] + [j]
            # Record for changed probability
            pre_prob = max_prob[:]
            # Record for new path
            path = new_path
        # Find the last state
        (prob, state) = max([(max_prob[i], i) for i in range(self.state_num)])
        
        return path[state]

    def _get_ob_index(self, observation):
        return [self._ob_list.index(i) for i in observation]  # Transform observation to index

    def _initial_ob_prob(self, ob_index):
        return [self._init_prob[i] * self._ob_prob[i][ob_index] for i in range(self.state_num)]

def main():
    hmm = HMM(3, ('up', 'down', 'unchanged'),
              initial_probability=[0.5, 0.2, 0.3],
              transition_probability=[[0.6, 0.2, 0.2],
                                      [0.5, 0.3, 0.2],
                                      [0.4, 0.1, 0.5]],
              observation_probability=[[0.7, 0.1, 0.2],
                                       [0.1, 0.6, 0.3],
                                       [0.3, 0.3, 0.4]])

    observation = ("up", "up", "unchanged", "down", "unchanged", "down", "up")
    ob_length = len(observation)
    p = hmm.forward(observation, ob_length)
    path = hmm.decode(observation, ob_length)
    print("P{} = {:.13f}".format(tuple(observation), p))
    print("Observation sequence =", tuple(i+1 for i in path))

if __name__ == '__main__':
    main()

