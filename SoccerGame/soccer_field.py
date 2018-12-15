import enum
from actions import Actions
from athlete import Athlete


class Soccer_Field:
    field_size = [4,2]
    # returns possible states for the game to be in given a player's actions

    def __init__(self, athlete1, athlete2):
        self.athlete1 = athlete1
        self.athlete2 = athlete2

    # these default methods need to be changed so == and != operations can be performed for custom classes.
    def __eq__(self, other):
        return self.athlete1 == other.athlete1 and self.athlete2 == other.athlete2

    def __ne__(self, other):
        return not self==other

    def __hash__(self):
        return hash((self.athlete1, self.athlete2))


    def get_reward(self):
        if self.athlete1.has_possession:
            x,y = self.athlete1.coordinates
        elif self.athlete2.has_possession:
            x,y = self.athlete2.coordinates

        if x == 0:
            return 100
        elif x == Soccer_Field.field_size[0]-1:
            return -100
        else:
            return 0

    def get_new_coordinates(self, athlete,action):
        #     0 1 2 3
        #   0 . . . .
        #   1 . . . .
        x,y=athlete.coordinates
        bottom_edge_of_field = 1
        right_edge_of_field = 3
        if action == Actions.down:
            y = min(y+1,bottom_edge_of_field)
        elif action == Actions.up:
            y = max(y-1, 0)
        elif action == Actions.left:
            x = max(x-1, 0)
        elif action == Actions.right:
            x = min(x+1, right_edge_of_field)
        return x,y

    def field_state_a2_first(self, athlete1_action, athlete2_action):

        a1_new_position = self.get_new_coordinates(self.athlete1, athlete1_action)
        a2_new_position = self.get_new_coordinates(self.athlete2, athlete2_action)


        if a1_new_position == a2_new_position:
            both_new_actions_are_the_same = True
        else:
            both_new_actions_are_the_same = False

        # What the field would look like if athlete 2 moved first
        if a2_new_position != self.athlete1.coordinates:
            if self.athlete2.has_possession:
                a2_state_if_a2_first = Athlete(coordinates=a2_new_position, has_possession=True, ID=2)
                if both_new_actions_are_the_same:
                    a1_state_if_a2_first = Athlete(coordinates=self.athlete1.coordinates, has_possession=False, ID=1)
                else:
                    a1_state_if_a2_first = Athlete(coordinates=a1_new_position, has_possession=False, ID=1)
            else:
                if both_new_actions_are_the_same:
                    a2_state_if_a2_first = Athlete(coordinates=a2_new_position, has_possession=True, ID=2)
                    a1_state_if_a2_first = Athlete(coordinates=self.athlete1.coordinates, has_possession=False, ID=1)
                else:
                    a2_state_if_a2_first = Athlete(coordinates=a2_new_position, has_possession=False, ID=2)
                    a1_state_if_a2_first = Athlete(coordinates=a1_new_position, has_possession=True, ID=1)
        else:   #if a2 moves to a1's coordinates..
            a2_state_if_a2_first = Athlete(coordinates=self.athlete2.coordinates, has_possession=False, ID=2)
            a1_state_if_a2_first = Athlete(coordinates=self.athlete1.coordinates, has_possession=True, ID=1)

        new_soccer_field = Soccer_Field(a1_state_if_a2_first, a2_state_if_a2_first)
        return new_soccer_field

    def field_state_a1_first(self, athlete1_action, athlete2_action):



        # update the soccer field if athlete 1 goes first
        a1_new_position = self.get_new_coordinates(self.athlete1, athlete1_action)
        a2_new_position = self.get_new_coordinates(self.athlete2, athlete2_action)

        if a1_new_position == a2_new_position:
            both_new_actions_are_the_same = True
        else:
            both_new_actions_are_the_same = False


        # What the field would look like if athlete 1 moved first
        '''if the player without the ball moves into the player with the ball, attempting to steal 
        the ball, he cannot.'''
        if a1_new_position != self.athlete2.coordinates:    #if a1's new move is NOT where a2 is...
            if self.athlete1.has_possession:                #AND a1 has possession...
                a1_state_if_a1_first = Athlete(coordinates=a1_new_position, has_possession=True, ID=1)   #then a1 can move anywhere
                if both_new_actions_are_the_same:
                    a2_state_if_a1_first = Athlete(coordinates=self.athlete2.coordinates, has_possession=False, ID=2) #a2 cannot move where a1 is because no possession
                else:
                    a2_state_if_a1_first = Athlete(coordinates=a2_new_position, has_possession=False, ID=2) #a2 can go anywhere as long as it's not where a1 is

            else:                                           #if a1 doesn't have the ball
                if both_new_actions_are_the_same:           #and both actions are the same...
                    a1_state_if_a1_first = Athlete(coordinates=a1_new_position, has_possession=True, ID=1) #a1 gets the ball
                    a2_state_if_a1_first = Athlete(coordinates=self.athlete2.coordinates, has_possession=False, ID=2) #a2 loses the ball
                else:
                    a1_state_if_a1_first = Athlete(coordinates=a1_new_position, has_possession=False, ID=1)
                    a2_state_if_a1_first = Athlete(coordinates=a2_new_position, has_possession=True, ID=2)
        #If the player with the ball moves into the player without it, the former loses the ball to the latter
        else:
            a1_state_if_a1_first = Athlete(coordinates=self.athlete1.coordinates, has_possession=False, ID=1)
            a2_state_if_a1_first = Athlete(coordinates=self.athlete2.coordinates, has_possession=True, ID=2)


        new_soccer_field = Soccer_Field(a1_state_if_a1_first, a2_state_if_a1_first)
        return new_soccer_field






