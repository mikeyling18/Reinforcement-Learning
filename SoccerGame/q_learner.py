from athlete import Athlete
from actions import Actions
from soccer_field import Soccer_Field
from collections import defaultdict
from match import Match
import copy as cp
import random as rd


class Q_Learner:
    def __init__(self):
        self.actions = [Actions.up, Actions.down, Actions.left, Actions.right, Actions.stick]
        self.initial_state = Soccer_Field(Athlete(coordinates = (2,0), has_possession=False, ID=1),
                                          Athlete(coordinates = (1,0), has_possession=True, ID = 2))

        self.V = defaultdict(lambda:1)

    def learn(self, total_t, alpha, gamma):
        print('Vanilla Q-Learner\n')
        Q_diff_list = list()

        Q = defaultdict(lambda:1)   # makes any new key's value = 1

        t = 0

        soccer_match = Match(self.initial_state)

        match_finished = False

        while t < total_t:

            if match_finished is True:
                match_finished = False
                soccer_match = Match(self.initial_state)
            t += 1
            S = cp.deepcopy(soccer_match.soccer_field)
            a1_action = rd.choice(self.actions)
            a2_action = rd.choice(self.actions)

            soccer_match.update_field(a1_action,a2_action)
            reward = soccer_match.soccer_field.get_reward()

            if reward is not 0:
                match_finished = True

            Q_max_check = list()

            for action in range(len(self.actions)):
                Q_max_check.append(Q[(soccer_match.soccer_field, action)])

            self.V[soccer_match.soccer_field] = max(Q_max_check)

            # we don't care about the other person's actions, so not included in Q
            q_before_update = Q[(S, a1_action)]
            Q[(S, a1_action)] = (1-alpha) * Q[(S, a1_action)] + alpha * (reward + gamma * self.V[soccer_match.soccer_field])
            q_after_update = Q[(S, a1_action)]

            # if (S, a1_action) == q_compare:
            if t % 1000 == 0:
                q_different = abs(q_after_update-q_before_update)
                Q_diff_list.append(q_different)
            if t % 10000 == 0:
                print('{} {}\n'.format(t,abs(q_after_update - q_before_update)))
            alpha = max(0.0001, alpha * 0.999999)

        return Q_diff_list