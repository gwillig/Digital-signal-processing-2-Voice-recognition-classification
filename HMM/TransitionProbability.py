import numpy as np


class TransitionProbability:
    def __init__(self, modelLength):
        self.modelLength = modelLength
        self.transProbs = np.zeros(shape=(modelLength, 3), dtype=np.float64)
        self.loops = np.zeros(shape=(modelLength), dtype=np.float64)
        self.nexts = np.zeros(shape=(modelLength), dtype=np.float64)
        self.skips = np.zeros(shape=(modelLength), dtype=np.float64)

    def incrementCounters(self, i, j):
        if i == j:
            self.loops[i] += 1
        if i == (j-1):
            self.nexts[i] += 1
        if i == (j-2):
            self.skips[i] += 1

    def calc(self):
        for j in range(0, self.modelLength):
            totalTransitions = self.loops[j] + self.nexts[j] + self.skips[j]

            self.transProbs[j][0] = self.loops[j] / totalTransitions
            self.transProbs[j][1] = self.nexts[j] / totalTransitions
            self.transProbs[j][2] = self.skips[j] / totalTransitions

        self.transProbs[self.transProbs < 0.01] = 0.01

    def resetCounters(self):
        self.loops = np.zeros(shape=(self.modelLength))
        self.nexts = np.zeros(shape=(self.modelLength))
        self.skips = np.zeros(shape=(self.modelLength))