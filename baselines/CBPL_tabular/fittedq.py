import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

class TabularFQI:
    def __init__(self, num_states, num_actions, max_epochs, gamma):
        '''
        An implementation of fitted Q iteration

        num_states: number of states
        num_actions: number of actions
        max_epochs: positive int, specifies how many iterations to run the algorithm
        gamma: discount factor
        '''
        self.max_epochs = max_epochs
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.ndarray


    def run(self,dataset,l):
        self.Q = np.random.rand(self.num_states,self.num_actions)
        X = dataset.get_state_action_pairs()
        x_prime = dataset.get_all('x_prime')
        rewards = dataset.get_combined_reward(l)
        #dones = dataset['done']

        for e in tqdm(range(self.max_epochs)):
            Y = rewards + self.gamma*np.max(self.Q[x_prime],axis=1)
            self.fit(X,Y)

        return TabularPolicy(self.Q,self.num_states,self.num_actions)


    def fit(self,X,Y):
        def cost(Q,X,Y,num_states,num_actions):
            Q = Q.reshape(num_states,num_actions)
            loss = 0
            for (x,a),y in zip(X,Y):
                loss += (Q[x,a]-y)**2
            return loss/len(Y)
        result = minimize(cost, self.Q.flatten(),args=(X,Y,self.num_states,self.num_actions),method='L-BFGS-B')
        self.Q = result.x.reshape(self.num_states,self.num_actions)

class TabularPolicy:
    def __init__(self, Q, num_states, num_actions):
        exp_Q = np.exp(np.array(Q))
        self.Q = exp_Q / np.sum(exp_Q,axis=1, keepdims=True)
        self.num_states = num_states
        self.num_actions = num_actions
    def __call__(self,states):
        return np.array([np.argmax(p) for p in self.Q[states]])


