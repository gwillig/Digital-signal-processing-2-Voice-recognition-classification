import os
from HMM.Viterbi import *
from HMM.TransitionProbability import TransitionProbability
import pickle
from tkinter import *

class Model:
    def __init__(self, folder, word, speaker, environment, modelLength = 0):
        self.alpha = 0
        self.word = word
        self.speaker = speaker
        self.environment = environment
        self.folder = folder
        self.featureVectorCount = []
        self.modelLength = modelLength
        self.importTrainingData()
        self.estimateModelAndVectorLength()
        self.iterations = "N/A"
        self.transProbs = TransitionProbability(self.modelLength)
        self.checkIfModelDataExist()

    def checkIfModelDataExist(self):
        if os.path.isfile(self.folder + "/TrainedModel/" + "mean_" + self.word + self.speaker + self.environment + ".csv") and os.path.isfile(
                          self.folder + "/TrainedModel/" + "variance_" + self.word + self.speaker + self.environment + ".csv") and os.path.isfile(
                          self.folder + "/TrainedModel/" + "transprobs_" + self.word + self.speaker + self.environment + ".csv") and os.path.isfile(
                          self.folder + "/TrainedModel/" + "info_" + self.word + self.speaker + self.environment + ".pkl"):
            self.mean = np.loadtxt(self.folder + "/TrainedModel/" + "mean_" + self.word + self.speaker + self.environment + ".csv", delimiter=" ")
            self.variance = np.loadtxt(self.folder + "/TrainedModel/" + "variance_" + self.word + self.speaker + self.environment + ".csv", delimiter=" ")
            self.transProbs.transProbs = np.loadtxt(self.folder + "/TrainedModel/" + "transprobs_" + self.word + self.speaker + self.environment + ".csv", delimiter=" ")
            with open(self.folder + "/TrainedModel/" + "info_" + self.word + self.speaker + self.environment + ".pkl",
                      'rb') as f:
                self.iterations = int(pickle.load(f))

    def printModelInformation(self):
        return "Model word: {0}\nModel Folder: {1}\nModel Feature Vector Count: {2}\nModel Length: {3}\nVector Length: {4}\nNumber of Iterations until convergence: {5}".format(self.word, self.folder, self.featureVectorCount, self.modelLength, self.vectorLength, self.iterations)

    def estimateModelAndVectorLength(self):
        if self.modelLength == 0:
            self.modelLength = int(np.median(self.featureVectorCount)/2)
        self.vectorLength = self.trainingData[0].shape[1]
        self.trainingDataCount = self.trainingData.shape[0]

    def importTrainingData(self):
        for root, dirs, files in os.walk(self.folder + "/Train/"):
            self.trainingData = np.empty(shape=len(files), dtype='object')
            for index, file in enumerate(files):
                self.trainingData[index] = np.genfromtxt(root + file, delimiter=' ')
                self.featureVectorCount.append(self.trainingData[index].shape[0])

    def train(self, infoBox):
        self.deleteModelFromDisc()
        self.mean, self.variance, self.counter = calculateInitialMeanAndVariance(self)
        infoBox.delete(1.0, END)
        infoBox.insert(END, "Training started...\n")
        infoBox.update()

        oldCounter = 0
        self.iterations = 0
        while not (self.counter == oldCounter).all():
            self.iterations += 1
            oldCounter = self.counter.copy()
            tempSum = 0
            tempSquaredSum = 0
            tempCounter = 0

            self.transProbs.resetCounters()
            # Do this with all available training instances
            for i in range(0, self.trainingDataCount):
                self.alpha, r, total = performViterbiAlgorithm(self.trainingData[i], self.mean, self.variance, self.transProbs.transProbs, self)
                x, y, z = trackBackwardPointer(self.trainingData[i], r, self)

                tempSum += x
                tempSquaredSum += y
                tempCounter += z

            self.transProbs.calc()
            self.mean = tempSum / tempCounter
            self.counter = tempCounter
            self.variance = tempSquaredSum / tempCounter - self.mean ** 2
            self.variance[self.variance < 0.01] = 0.01

            infoBox.insert(END, "Iteration #{0}\n".format(str(self.iterations)))
            infoBox.update()
            infoBox.yview_moveto(1)

        infoBox.insert(END, "Training Completed ...\nNumber of Iterations until convergence: {0}".format(self.iterations))
        infoBox.update()
        self.saveModelToDisc()


    def deleteModelFromDisc(self):
        # Delete Array if it already exists
        if os.path.isfile(self.folder + "/TrainedModel/" + "mean_" + self.word + self.speaker + self.environment + ".csv"):
            os.remove(self.folder + "/TrainedModel/" + "mean_" + self.word + self.speaker + self.environment + ".csv")
        if os.path.isfile(self.folder + "/TrainedModel/" + "variance_" + self.word + self.speaker + self.environment + ".csv"):
            os.remove(self.folder + "/TrainedModel/" + "variance_" + self.word + self.speaker + self.environment + ".csv")
        if os.path.isfile(self.folder + "/TrainedModel/" + "transprobs_" + self.word + self.speaker + self.environment + ".csv"):
            os.remove(self.folder + "/TrainedModel/" + "transprobs_" + self.word + self.speaker + self.environment + ".csv")
        if os.path.isfile(self.folder + "/TrainedModel/" + "info_" + self.word + self.speaker + self.environment + ".pkl"):
            os.remove(self.folder + "/TrainedModel/" + "info_" + self.word + self.speaker + self.environment + ".pkl")

        # reset Arrays
        self.mean = np.empty((self.modelLength, self.vectorLength))
        self.variance = np.empty((self.modelLength, self.vectorLength))
        self.transProbs.transProbs = np.empty((self.modelLength, 3))


    def saveModelToDisc(self):
        # Save it to disc
        np.savetxt(self.folder + "/TrainedModel/" + "mean_" + self.word + self.speaker + self.environment + ".csv", self.mean, delimiter=" ")
        np.savetxt(self.folder + "/TrainedModel/" + "variance_" + self.word + self.speaker + self.environment + ".csv", self.variance, delimiter=" ")
        np.savetxt(self.folder + "/TrainedModel/" + "transprobs_" + self.word + self.speaker + self.environment + ".csv", self.transProbs.transProbs, delimiter=" ")
        with open(self.folder + "/TrainedModel/" + "info_" + self.word + self.speaker + self.environment + ".pkl", 'wb') as f:
            pickle.dump(self.iterations, f)

