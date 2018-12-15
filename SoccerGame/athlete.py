
class Athlete:
    def __init__(self, coordinates, has_possession, ID):
        self.coordinates = coordinates
        self.has_possession = has_possession
        self.ID = ID


    # these default methods need to be changed so == and != operations can be performed for custom classes.
    def __eq__(self, other):
        return self.ID == other.ID and self.coordinates == other.coordinates \
               and self.has_possession == other.has_possession

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.ID, self.coordinates, self.has_possession))
