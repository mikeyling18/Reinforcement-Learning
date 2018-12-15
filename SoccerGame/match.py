import random as rd
import copy as cp

class Match:

    def __init__(self, soccer_field):
        self.soccer_field = cp.deepcopy(soccer_field)

    def update_field(self,a1_action, a2_action):
        j = rd.randint(1, 2)
        if j == 1:
            self.soccer_field = self.soccer_field.field_state_a1_first(a1_action, a2_action)
        else:
            self.soccer_field = self.soccer_field.field_state_a2_first(a1_action, a2_action)
