from hmm import HMM


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
    print("P{} = {:.13f}".format(tuple(observation), hmm.backward(observation, ob_length)))
    print("Observation sequence =", tuple(i+1 for i in path))

if __name__ == '__main__':
    main()
