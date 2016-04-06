class HMM:
    """Simple implement for Hidden Markov Model"""
    def __init__(self, state_num, observation_list):
        self.state_num = state_num
        self.init_prob = [0 for i in range(state_num)]                                  # Initial probability for choosing first state
        self.state = [i for i in range(state_num)]
        self.state_prob = [[0 for j in range(state_num)] for i in range(state_num)]     # Every state's transition probability
        self.ob_list = observation_list
        self.ob_num = len(observation_list)
        self.ob_prob = [[0 for j in range(self.ob_num)] for i in range(self.state_num)] # Every state's observation probability

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
        prob_list = self._initial_ob_prob(ob_list[0])
        pre_prob = prob_list[:]

        for t in range(1, time):
            for j in range(self.state_num):
                # Calculate probability that previous state probability transit to present state probability
                p = sum([pre_prob[i] * self.state_prob[i][j] for i in range(self.state_num)])
                # Calculate probability that present state to present observation
                prob_list[j] = p * self.ob_prob[j][ob_list[t]]
            # Record for changed probability
            pre_prob = prob_list[:]
        
        return sum(prob_list)

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
                p, state = max([(pre_prob[i] * self.state_prob[i][j], i) for i in range(self.state_num)])
                # Calculate probability that present state to present observation
                max_prob[j] = p * self.ob_prob[j][ob_list[t]]
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
        return [self.ob_list.index(i) for i in observation]  # Transform observation to index

    def _initial_ob_prob(self, ob_index):
        return [self.init_prob[i] * self.ob_prob[i][ob_index] for i in range(self.state_num)]

def main():
    hmm = HMM(3, ('up', 'down', 'unchanged'))
    hmm.init_prob = [0.5, 0.2, 0.3]
    hmm.state_prob = [[0.6, 0.2, 0.2],
                      [0.5, 0.3, 0.2],
                      [0.4, 0.1, 0.5]]
    hmm.ob_prob = [[0.7, 0.1, 0.2],
                   [0.1, 0.6, 0.3],
                   [0.3, 0.3, 0.4]]
    observation = ("up", "up", "unchanged", "down", "unchanged", "down", "up")
    ob_length = len(observation)
    p = hmm.forward(observation, ob_length)
    path = hmm.decode(observation, ob_length)
    print("P{} = {:.13f}".format(tuple(observation), p))
    print("Observation sequence =", tuple(i+1 for i in path))

if __name__ == '__main__':
    main()