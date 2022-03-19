from model_player import player_model

class attacker_model(player_model):
    def __init__(self, system):
        player_model.__init__(self, system)
        self.maximum_signal_radius_HD = 3   # signal radius of HD
        self.sg_HD = 5                      # defense strategy, range [0, 10]



