import numpy as np
import pickle
import os

class Dataset:
    def __init__(self, capacity=3000000):
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._end = 0
        self._start = 0
        self._size = 0

    def __len__(self):
        return self._size

    def __setitem__(self, index, data):
        assert index < self._size and index >= -self._size
        self._buffer[(index + self._start) % self.capacity] = data

    def __getitem__(self, index):
        assert index < self._size and index >= -self._size
        return self._buffer[(index + self._start) % self.capacity]

    def clear(self):
        self._buffer = [None] * self.capacity
        self._end = 0
        self._start = 0
        self._size = 0

    def append(self, data):
        self._buffer[self._end] = data
        if self._size == self.capacity:
            self._start = (self._start + 1) % self.capacity
        else:
            self._size += 1
        self._end = (self._end + 1) % self.capacity

    def sample(self, batch_size):
        assert self._size >= batch_size
        indices = np.random.choice(self._size, batch_size, replace=False)

        return [self.__getitem__(index) for index in indices]

    def load_from_trajectories(self,trajectories):
        for trajectory in trajectories:
            for t in trajectory:
                self.append(t)
        print("Dataset size: ", self._size)

    def load_from_files(self,folder):
        def load_trajectories(file_path):
            with open(file_path, "rb") as f:
                trajectories = pickle.load(f)
            return trajectories

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            trajectories = load_trajectories(file_path)
            for trajectory in trajectories:
                states = trajectory['state']
                actions = trajectory['action']
                rewards = trajectory['reward']
                costs = trajectory['cost']
                next_states = trajectory['next_state']
                dones = trajectory['done']
                initial_states = trajectory['initial_state']

                for state,action,reward,cost,next_state,done,initial_state in \
                        zip(states,actions,rewards,costs,next_states,dones,initial_states):
                    self.append((state,action,reward,cost,next_state,done,initial_state))

            print("load offline dataset: {}".format(file_path))


