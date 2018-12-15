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


class Correlated_Learner:
    def __init__(self):
        self.actions = [Actions.up, Actions.down, Actions.left, Actions.right, Actions.stick]
        self.initial_state = Soccer_Field(Athlete(coordinates = (2,0), has_possession=False, ID = 1),
                                          Athlete(coordinates = (1,0), has_possession=True, ID = 2))

        self.Value1 = defaultdict(lambda:1)
        self.Value2 = defaultdict(lambda:1)

    def learn(self, total_t, alpha, gamma):
        print('Correlated Q-Learner\n')
        options['show_progress'] = False
        Q_diff_list = list()

        Q_a1 = defaultdict(lambda:1)   # makes any new key's value = 1
        Q_a2 = defaultdict(lambda:1)
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
            # of each action, pi(a1,a2)

            # create list of cvxopt.modeling constraints to use for solver
            # Now we have 25 probabilities to take into account because if each athlete has 5 moves
            # So we need to keep track of the probabilities of each athlete 1 action-athlete 2 action combination

            P = {}
            C = list()
            for action_set1 in range(len(self.actions)):
                for action_set2 in range(len(self.actions)):
                    P[(action_set1, action_set2)] = variable()
                    C.append(P[(action_set1, action_set2)] >= 0)

            sum_all_P = 0
            # All 25 a1-action, a2-action probabilities need to add (sum) to one
            for action_set1 in range(len(self.actions)):
                for action_set2 in range(len(self.actions)):
                    sum_all_P += P[(action_set1, action_set2)]
            C.append(sum_all_P == 1)

            total_value = 0
            for action_set1 in range(len(self.actions)):
                for action_set2 in range(len(self.actions)):
                    total_value += P[(action_set1, action_set2)] * Q_a1[(soccer_match.soccer_field, self.actions[action_set1], self.actions[action_set2])]
                    total_value += P[(action_set1, action_set2)] * Q_a2[(soccer_match.soccer_field, self.actions[action_set1], self.actions[action_set2])]
            C.append(Nash_i >= total_value)

            lp = op(-Nash_i, C)
            lp.solve()

            self.Value1[soccer_match.soccer_field] = Nash_i.value[0]
            self.Value2[soccer_match.soccer_field] = Nash_i.value[0]

            # we don't care about the other person's actions, so not included in Q
            q_before_update = Q_a1[S, a1_action, a2_action]
            Q_a1[(S, a1_action, a2_action)] = (1-alpha) * Q_a1[(S, a1_action, a2_action)] + alpha * (reward + gamma * self.Value1[soccer_match.soccer_field])
            Q_a2[(S, a1_action, a2_action)] = (1-alpha) * Q_a2[(S, a1_action, a2_action)] + alpha * (reward + gamma * self.Value2[soccer_match.soccer_field])
            q_after_update = Q_a1[(S, a1_action, a2_action)]
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