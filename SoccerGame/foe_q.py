from athlete import Athlete
from actions import Actions
from soccer_field import Soccer_Field
from collections import defaultdict
from match import Match
import copy as cp
import random as rd

from cvxopt.modeling import op
from cvxopt.modeling import variable
from cvxopt.solvers import options


class Foe_Learner:
    def __init__(self):
        self.actions = [Actions.up, Actions.down, Actions.left, Actions.right, Actions.stick]
        self.initial_state = Soccer_Field(Athlete(coordinates = (2,0), has_possession=False, ID=1),
                                          Athlete(coordinates = (1,0), has_possession=True, ID = 2))

        self.V = defaultdict(lambda:1)

    def learn(self, total_t, alpha, gamma):
        print('Foe Q-Learner\n')
        Q_diff_list = list()

        Q = defaultdict(lambda: 1)   # makes any new key's value = 1

        t = 0

        soccer_match = Match(self.initial_state)

        match_finished = False

        while t < total_t:

            if match_finished is True:
                match_finished = False
                soccer_match = Match(self.initial_state)
            t += 1

            a1_action = rd.choice(self.actions)
            a2_action = rd.choice(self.actions)

            S = cp.deepcopy(soccer_match.soccer_field)
            soccer_match.update_field(a1_action, a2_action)
            reward = soccer_match.soccer_field.get_reward()


            if reward is not 0:
                match_finished = True

            Nash_i = variable()

            # create a list of cvxopt.modeling variables to represent the probability
            # of each action, pi(a1) in the paper
            P = list()
            for available_actions in range(len(self.actions)):
                P.append(variable())

            # create list of cvxopt.modeling constraints to use for solver
            C = list()
            for available_actions in range(len(self.actions)):
                C.append(P[available_actions] >= 0)

            C.append(sum(P) == 1)

            for ones_actions in range(len(self.actions)):
                x = 0
                for anothers_actions in range(len(self.actions)):
                    # Equation 8 in Littman's "Friend or Foe Q-Learning in General-Sum Games"
                    x += P[anothers_actions] * Q[(soccer_match.soccer_field, self.actions[anothers_actions], self.actions[ones_actions])]
                C.append(x >= Nash_i)

            lp = op(-Nash_i, C)
            lp.solve()

            self.V[soccer_match.soccer_field] = Nash_i.value[0]

            # we don't care about the other person's actions, so not included in Q
            q_before_update = Q[S, a1_action, a2_action]
            Q[(S, a1_action, a2_action)] = (1-alpha) * Q[(S, a1_action, a2_action)] + alpha * (reward + gamma * self.V[soccer_match.soccer_field])
            q_after_update = Q[(S, a1_action, a2_action)]
            q_different = abs(q_after_update - q_before_update)

            if S.athlete1.coordinates == (2,0) and S.athlete2.coordinates == (1,0) and \
                S.athlete1.has_possession == False and S.athlete2.has_possession == True and \
                a1_action == Actions.down and a2_action == Actions.stick:
                Q_diff_list.append(q_different)
                print('{} {}\n'.format(t, abs(q_after_update - q_before_update)))
            # if t % 10000 == 0:
            #     print('{} {}\n'.format(t, abs(q_after_update - q_before_update)))
            alpha = max(0.001, alpha * 0.999999)

        return Q_diff_list