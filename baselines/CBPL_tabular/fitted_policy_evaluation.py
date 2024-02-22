import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

class TabularFQE:
    def __init__(self, initial_states,num_states, num_actions, max_epochs, gamma):

        '''
        An implementation of fitted Q iteration

        dim_of_actions: dimension of action space
        max_epochs: positive int, specifies how many iterations to run the algorithm
        gamma: discount factor
        '''
        self.initial_states = initial_states
        self.num_states = num_states
        self.num_actions = num_actions
        self.max_epochs = max_epochs
        self.gamma = gamma

    def run(self,policy, which_value, dataset):
        self.Q = np.random.rand(self.num_states, self.num_actions)
        X = dataset.get_state_action_pairs()
        x_prime = dataset.get_all('x_prime')
        V = np.squeeze(dataset.get_all(which_value))
        actions = policy(x_prime)

        values = []
        for e in tqdm(range(self.max_epochs)):
            Y = V + self.gamma*self.Q[x_prime,actions]
            self.fit(X,Y)
            values.append(np.mean(self.Q[self.initial_states, policy(self.initial_states)]))

        return np.mean(values[-5:]), values

    def fit(self,X,Y):
        def cost(Q,X,Y,num_states,num_actions):
            Q = Q.reshape(num_states,num_actions)
            loss = 0
            for (x,a),y in zip(X,Y):
                loss += (Q[x,a]-y)**2
            return loss/len(Y)
        result = minimize(cost, self.Q.flatten(),args=(X,Y,self.num_states,self.num_actions),method='L-BFGS-B')
        self.Q = result.x.reshape(self.num_states,self.num_actions)
