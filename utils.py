import numpy as np


class Perceptron:

    def __init__(self, w1=0, w2=0, wb=0):
        self.w1 = w1
        self.w2 = w2
        self.wb = wb

    def train(self, training_sets, targets):
        e = np.zeros((len(training_sets)))

        while True:

            for i, (point, t) in enumerate(zip(training_sets, targets)):
                y = self.predict(point)
                e[i] = t - y
                self.w1 = self.w1 + point[0] * e[i]
                self.w2 = self.w2 + point[1] * e[i]
                self.wb = self.wb + e[i]

            if any(e) is False:
                break

    def predict(self, point):
        a = self.w1 * point[0] + self.w2 * point[1] + self.wb
        return 1.0 if a > 0.0 else 0.0
