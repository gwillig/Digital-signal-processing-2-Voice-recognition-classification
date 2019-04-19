import numpy as np

def calculateInitialMeanAndVariance(m):
    # Calculate Initial Gradient
    initialGradient = np.zeros(shape=m.trainingDataCount)
    for i, content in enumerate(m.trainingData):
        initialGradient[i] = (m.modelLength - 1) / (content.shape[0] - 1)

    # Initalize Variables
    counter = np.zeros(shape=(m.modelLength, 1))
    sum = np.zeros(shape=(m.modelLength, m.vectorLength))
    squaredSum = np.zeros(shape=(m.modelLength, m.vectorLength))

    stateOld = 0


    for i, content in enumerate(m.trainingData):

        for j, x in enumerate(content):
            state = int(np.round(initialGradient[i] * j))
            if j!=0:
                m.transProbs.incrementCounters(stateOld, state)
            stateOld = state

            counter[state] += 1
            sum[state] += x
            squaredSum[state] += x ** 2

    m.transProbs.calc()


    mean = np.divide(sum, counter)
    variance = squaredSum / counter - mean ** 2
    variance[variance < 0.01] = 0.01

    return mean, variance, counter

def concatModels(silenceModel, models):
    bigMean = silenceModel.mean
    bigVariance = silenceModel.variance
    bigTransprobs = silenceModel.transProbs.transProbs

    modelIndexStart = [0]
    modelIndexEnd = [0]
    wordArray = ["Stille"]
    speakerArray = ["None"]
    environmentArray = ["None"]

    for i, model in enumerate(models):

        bigMean = np.vstack((bigMean, model.mean))
        bigVariance = np.vstack((bigVariance, model.variance))
        bigTransprobs = np.vstack((bigTransprobs, model.transProbs.transProbs))

        modelIndexStart.append(modelIndexEnd[-1]+1)
        modelIndexEnd.append(modelIndexEnd[-1] + model.modelLength)
        wordArray.append(model.word)
        speakerArray.append(model.speaker)
        environmentArray.append(model.environment)

    return bigMean, bigVariance, bigTransprobs, modelIndexStart, modelIndexEnd, wordArray, speakerArray, environmentArray

def performViterbiForPrediction(ref, bigMean, bigVariance, bigTransProbs, modelIndexStart, modelIndexEnd):
    # init: alpha
    alpha = np.zeros(shape=(len(ref), len(bigMean)))
    # init: backward pointer r
    r = np.zeros(shape=(len(ref), len(bigMean)))
    # initalize first column except of first node with infinity
    for j in range(0, len(bigMean)):
        alpha[0][j] = float('Inf')
    # init: first node
    #for x in modelIndexStart:
    alpha[0][0] = calculateNormalDistributionDistance(0, 0, ref, bigMean[0], bigVariance[0])

    # traverse trellis diagram
    for t in range(1, len(ref)):
        # Stille
        alpha_min = np.amin(alpha[t - 1, modelIndexEnd])
        previous_state = np.argmin(alpha[t - 1, modelIndexEnd])
        alpha[t][0] = alpha_min + calculateNormalDistributionDistance(t, 0, ref, bigMean, bigVariance)
        r[t][0] = modelIndexEnd[previous_state]

        for state in range(1, len(bigMean)):
            # Special Case State 0: means only loop was possible

            if state in modelIndexStart:
                alpha_min = alpha[t - 1][0]
                previous_state = 0
            # Special Case State 1: means only loop or next was possible

            elif state-1 in modelIndexStart:
                alpha_min = np.amin(alpha[t - 1][state-1:state+1] - np.log(np.diag(np.flipud((bigTransProbs[state-1: state+1, 0:2])))[::-1]))
                previous_state = np.argmin(alpha[t - 1][state-1:state+1] - np.log(np.diag(np.flipud((bigTransProbs[state-1:state+1, 0:2])))[::-1])) -1 + state
            # Special Case State >=2: means loop or next or skip was possible

            else:
                alpha_min = np.amin(alpha[t - 1][state - 2:state + 1] - np.log(np.diag(np.flipud((bigTransProbs[state - 2:state + 1, :])))[::-1]))
                previous_state = np.argmin(alpha[t - 1][state - 2:state + 1] - np.log(np.diag(np.flipud((bigTransProbs[state - 2:state + 1, :])))[::-1])) - 2 + state

            alpha[t][state] = alpha_min + calculateNormalDistributionDistance(t, state, ref, bigMean, bigVariance)
            r[t][state] = previous_state
    return r, alpha

def trackBackwardPointerForPrediction(ref, r, words, speakers, environments, ends, costMatrix):
    state = 0
    outputWord = []
    outputSpeaker = []
    outputCost = []
    outputEnvironment = []
    for i in range(len(ref) - 1, -1, -1):
        if state == 0:
            if int(r[i][state]) in ends:
                x = ends.index(int(r[i][state]))
                if x != 0:
                    outputWord.append(words[x])
                    outputSpeaker.append(speakers[x])
                    outputCost.append(costMatrix[i][int(r[i][state])])
                    outputEnvironment.append((environments[x]))


        state = int(r[i][state])
    return outputWord[::-1], outputSpeaker[::-1], outputCost[::-1], outputEnvironment[::-1] # Reverse Output

def calculateNormalDistributionDistance(t, state, ref, mean, variance):
    return 1/2*np.sum(np.log(2*np.pi*variance[state]) + ((ref[t] - mean[state])**2) / variance[state])

def calculateMahalanobisDistance(t, state, ref, mean, variance):
    return np.sum((ref[t] - mean[state])**2 / variance[state])

def performViterbiAlgorithm(ref, mean, variance, transProbs, m):
    # init: alpha
    alpha = np.zeros(shape=(len(ref), len(mean)))
    # init: backward pointer r
    r = np.zeros(shape=(len(ref), len(mean)))
    # init: first node
    alpha[0][0] = calculateNormalDistributionDistance(0, 0, ref, mean, variance)

    # initalize first column except of first node with infinity
    for j in range(1, len(mean)):
        alpha[0][j] = float('Inf')

    # traverse trellis diagram
    for t in range(1, len(ref)):
        for state in range(0, m.modelLength):
            # Special Case State 0: means only loop was possible
            if state == 0:
                alpha_min = np.amin(alpha[t - 1][0] - np.log(transProbs[0][0]))
                previous_state = 0

            # Special Case State 1: means only loop or next was possible
            elif state == 1:
                alpha_min = np.amin(alpha[t - 1][0:2] - np.log(np.diag(np.flipud((transProbs[0:2,0:2])))[::-1]))
                previous_state = np.argmin(alpha[t - 1][0:2] - np.log(np.diag(np.flipud((transProbs[0:2,0:2])))[::-1]))

            # Special Case State >=2: means loop or next or skip was possible
            else:
                alpha_min = np.amin(alpha[t - 1][state - 2:state + 1] - np.log(np.diag(np.flipud((transProbs[state-2:state+1,:])))[::-1]))
                previous_state = np.argmin(alpha[t - 1][state - 2:state + 1] - np.log(np.diag(np.flipud((transProbs[state-2:state+1,:])))[::-1])) - 2 + state

            alpha[t][state] = alpha_min + calculateNormalDistributionDistance(t, state, ref, mean, variance)
            r[t][state] = previous_state


    total = alpha[-1][-1] # Last Element
    return alpha, r, total


def trackBackwardPointer(ref, r, m):
    state = r.shape[1] - 1

    summe = np.zeros(shape=(m.modelLength, m.vectorLength))
    squaredSumme = np.zeros(shape=(m.modelLength, m.vectorLength))
    counter = np.zeros(shape=(m.modelLength, 1))

    for i in range(len(ref) - 1, -1, -1):

        summe[state] += ref[i]
        squaredSumme[state] += ref[i] ** 2
        counter[state] += 1

        if i != 0:
            m.transProbs.incrementCounters(int(r[i][state]), state)

        state = int(r[i][state])

    return summe, squaredSumme, counter