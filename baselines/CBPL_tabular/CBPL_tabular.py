from MDP_util import *
from fittedq import TabularPolicy


class CBPL:
    def __init__(self, dataset, constraints, n_states, n_actions, best_response_algorithm, online_convex_algorithm,
                 fitted_off_policy_evaluation_algorithm, lambda_bound=1., epsilon=.01):
        '''
        This is a problem of the form: max_pi R(pi) where C(pi) < eta.

        dataset: list. Will be {(x,a,x',r(x,a), c(x,a)^T)}
        action_space_dim: number of dimension of action space
        dim: number of constraints
        best_response_algorithm: function which accepts a |A| dim vector and outputs a policy which minimizes L
        online_convex_algorithm: function which accepts a policy and returns an |A| dim vector (lambda) which maximizes L
        lambda_bound: positive int. l1 bound on lambda |lambda|_1 <= B
        constraints:  |A| dim vector
        epsilon: small positive float. Denotes when this problem has been solved.
        '''

        self.dataset = dataset
        self.constraints = constraints
        self.R = []
        self.C = []
        self.policies = []
        self.n_actions = n_actions
        self.n_states = n_states
        self.dim = len(constraints)
        self.lambda_bound = lambda_bound
        self.epsilon = epsilon
        self.best_response_algorithm = best_response_algorithm
        self.online_convex_algorithm = online_convex_algorithm
        self.exact_lambdas = []
        self.fitted_off_policy_evaluation_algorithm = fitted_off_policy_evaluation_algorithm

    def best_response(self, lamb):
        '''
        Best-response(lambda) = argmin_{pi} L(pi, lambda)
        '''
        policy = self.best_response_algorithm.run(self.dataset, lamb)
        return policy

    def online_algo(self):
        '''
        No regret online convex optimization routine
        '''
        # Gradient = 0 for dummy lambda
        gradient = self.constraints - self.C[-1]
        lambda_t = self.online_convex_algorithm.run(gradient)

        return lambda_t

    def lagrangian(self, R, C, lamb):
        # R(pi) + lambda^T (tau - C(pi)), where tau = constraints, pi = avg of all pi's seen
        return R + np.dot(lamb, (self.constraints - C))

    def min_of_lagrangian_over_lambda(self):
        '''
        The minimum of R(pi) + lambda^T (tau - C(pi)).
        '''
        lamb = np.array([self.lambda_bound])
        # Add a dummy lambda of 0
        return min(self.lagrangian(np.mean(self.R), np.mean(self.C), lamb), np.mean(self.R))

    def max_of_lagrangian_over_policy(self, lambdas):
        '''
        This function evaluates L(best_response(avg_lambda), avg_lambda)
        '''

        lamb_avg = np.mean(lambdas)
        best_policy = self.best_response([lamb_avg])
        # R(best_response(lambda_avg))
        R_br, values = self.fitted_off_policy_evaluation_algorithm.run(best_policy, 'r', self.dataset)

        # C(best_response(lambda_avg))
        C_br, values = self.fitted_off_policy_evaluation_algorithm.run(best_policy, 'c', self.dataset)

        return self.lagrangian(R_br,C_br,lamb_avg), R_br, C_br

    def update(self, policy):

        # update R
        R_pi, eval_values = self.fitted_off_policy_evaluation_algorithm.run(policy, 'r', self.dataset)
        self.R.append(R_pi)
        self.policies.append(policy)

        # update C
        C_pi, eval_values = self.fitted_off_policy_evaluation_algorithm.run(policy, 'c', self.dataset)
        self.C.append(C_pi)


    def is_over(self, lambdas):
        # lambdas: list. We care about average of all lambdas seen thus far
        # If |max_lambda L(avg_pi, lambda) - L(best_response(avg_lambda), avg_lambda)| < epsilon, then done

        if len(lambdas) <= 1: return False, 0

        else:
            x = self.min_of_lagrangian_over_lambda()
            y, r_br, c_br = self.max_of_lagrangian_over_policy(lambdas)

        difference = y - x

        r_approx, c_approx = np.mean(self.R), np.mean(self.C)

        if difference < self.epsilon:
            return True, difference
        else:
            return False, difference

    def get_policy(self):
        return self.policies[-1]


