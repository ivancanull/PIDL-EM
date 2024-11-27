class Normalizer():
    def __init__(self):
        self.mean = []
        self.std = []

    def normalize(self, x):
        x = (x - self.mean) / self.std
        return x
