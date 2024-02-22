import numpy as np


class Dataset:
    def __init__(self, capacity):
        self.capacity = capacity
        # x, a, x_prime, r, c
        self.data = {'x': [None] * capacity, 'a': [None] * capacity, 'x_prime': [None] * capacity,
                     'r': [None] * capacity, 'c': [None] * capacity}
        self._end = 0
        self._start = 0
        self._size = 0

    def __len__(self):
        return self._size

    def __setitem__(self, index, data):
        assert index < self._size and index >= -self._size
        index = (index + self._start) % self.capacity
        self.data['x'][index] = data['x']
        self.data['a'][index] = data['a']
        self.data['x_prime'][index] = data['x_prime']
        self.data['r'][index] = data['r']
        self.data['c'][index] = data['c']

    def __getitem__(self, index):
        assert index < self._size and index >= -self._size
        index = (index + self._start) % self.capacity
        x = self.data['x'][index]
        a=self.data['a'][index]
        x_prime=self.data['x_prime'][index]
        r=self.data['r'][index]
        c=self.data['c'][index]
        experience = [x,a,x_prime,r,c]
        return experience


    def clear(self):
        self.data = {'x': [None] * self.capacity, 'a': [None] * self.capacity, 'x_prime': [None] * self.capacity,
                     'r': [None] * self.capacity, 'c': [None] * self.capacity}
        self._end = 0
        self._start = 0
        self._size = 0

    def append(self,data):
        if self._size == self.capacity:
            self._start = (self._start + 1) % self.capacity
        else:
            self._size += 1
        self.__setitem__(self._end,data)

        self._end = (self._end + 1) % self.capacity

    def sample(self,batch_size):
        assert self._size >= batch_size
        indices = np.random.choice(self._size, batch_size, replace=False)
        return [self.__getitem__(index) for index in indices]

    def get_all(self,key):
        if self._size == self.capacity:
            return np.array(self.data[key])
        else:
            return np.array(self.data[key][:self._end])

    def get_combined_reward(self,l):
        r = np.array(self.get_all('r'))
        c = np.array(self.get_all('c'))
        l = np.array(l)
        return r - np.dot(c,l.T)

    def get_state_action_pairs(self):
        states = self.get_all('x')
        actions = self.get_all('a')
        return np.array([states,actions]).T

