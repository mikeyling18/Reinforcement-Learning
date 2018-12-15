from athlete import Athlete
from actions import Actions
from soccer_field import Soccer_Field
from collections import defaultdict
from match import Match

import copy as cp
import random as rd



class Friend_Learner:
    def __init__(self):
        self.actions = [Actions.up, Actions.down, Actions.right, Actions.left, Actions.stick]
        self.initial_state = Soccer_Field(Athlete(coordinates = (2,0), has_possession=False, ID=1),
                                          Athlete(coordinates = (1,0), has_possession=True, ID = 2))
        self.Nash = defaultdict(lambda: 1)



    def learn(self, total_t, alpha, gamma):
        print('FRIEND Q-LEARNER\n')
        Q_diff_list = list()

        Q = defaultdict(lambda: 1)   # makes any new key's value = 1

        t = 0

        soccer_match = Match(self.initial_state)
        a1_coords1 = soccer_match.soccer_field.athlete1.coordinates
        a2_coords1 = soccer_match.soccer_field.athlete2.coordinates
        match_finished = False

        while t < total_t:

            if match_finished is True:
                match_finished = False
                soccer_match = Match(self.initial_state)
            t += 1

            a1_action = rd.choice(self.actions)
            a2_action = rd.choice(self.actions)

            S = cp.deepcopy(soccer_match.soccer_field)

            a1_ball1 = S.athlete1.has_possession
            a2_ball2 = S.athlete2.has_possession
            a1_coords2 = soccer_match.soccer_field.athlete1.coordinates
            a2_coords2 = soccer_match.soccer_field.athlete2.coordinates
            soccer_match.update_field(a1_action, a2_action)
            a1_ball = S.athlete1.has_possession
            a2_ball = S.athlete2.has_possession
            a1_coords = soccer_match.soccer_field.athlete1.coordinates
            a2_coords = soccer_match.soccer_field.athlete2.coordinates
            reward = soccer_match.soccer_field.get_reward()

            if reward != 0:
                match_finished = True

            nash_q = Q[(soccer_match.soccer_field, Actions.up, Actions.up)]

            # Equation 7 from Littmans FFQ Paper
            for a in self.actions:
                for o in self.actions:
                    nash_q = max(nash_q, Q[(soccer_match.soccer_field, a, o)])

            self.Nash[soccer_match.soccer_field] = nash_q

            # we don't care about the other person's actions, so not included in Q
            q_before_update = Q[(S, a1_action, a2_action)]
            tmp = ((1-alpha) * Q[(S, a1_action, a2_action)]) + (alpha * (reward + gamma * self.Nash[soccer_match.soccer_field]))
            Q[(S, a1_action, a2_action)] = tmp
            q_after_update = Q[(S, a1_action, a2_action)]
            q_different = abs(q_after_update - q_before_update)

            if S.athlete1.coordinates == (2,0) and S.athlete2.coordinates == (1,0) and \
                S.athlete1.has_possession == False and S.athlete2.has_possession == True and \
                a1_action == Actions.down and a2_action == Actions.stick:
                Q_diff_list.append(q_different)

            if t % 10000 == 0:
                print('{} {}\n'.format(t, q_different))
            alpha = max(0.001, alpha * 0.999999)

        return Q_diff_list