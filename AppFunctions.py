from HMM.Model import Model
import numpy as np

def loadModels():
    modelList = np.array([
        Model("./data/TrainingData/A409/Silence/", word="Silence", speaker="None", environment="A409", modelLength=1),
        Model("./data/TrainingData/G117/Silence/", word="Silence", speaker="None", environment="G117", modelLength=1),

        Model("./data/TrainingData/A409/Zwei/Andreas/", word="Zwei", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Zwei/Gustav/", word="Zwei", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Zwei/Lisa/", word="Zwei", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Zwei/All/", word="Zwei", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Drei/Andreas/", word="Drei", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Drei/Gustav/", word="Drei", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Drei/Lisa/", word="Drei", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Drei/All/", word="Drei", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Lisa/Andreas/", word="Lisa", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Lisa/Gustav/", word="Lisa", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Lisa/Lisa/", word="Lisa", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Lisa/All/", word="Lisa", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Gustav/Andreas/", word="Gustav", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Gustav/Gustav/", word="Gustav", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Gustav/Lisa/", word="Gustav", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Gustav/All/", word="Gustav", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Andreas/Andreas/", word="Andreas", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Andreas/Gustav/", word="Andreas", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Andreas/Lisa/", word="Andreas", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Andreas/All/", word="Andreas", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Und/Andreas/", word="Und", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Und/Gustav/", word="Und", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Und/Lisa/", word="Und", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Und/All/", word="Und", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Moegen/Andreas/", word="Moegen", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Moegen/Gustav/", word="Moegen", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Moegen/Lisa/", word="Moegen", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Moegen/All/", word="Moegen", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Signalverarbeitung/Andreas/", word="Signalverarbeitung", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Signalverarbeitung/Gustav/", word="Signalverarbeitung", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Signalverarbeitung/Lisa/", word="Signalverarbeitung", speaker="Lisa", environment="A409"),
        Model("./data/TrainingData/A409/Signalverarbeitung/All/", word="Signalverarbeitung", speaker="All", environment="A409"),

        Model("./data/TrainingData/A409/Fouriertransformation/Andreas/", word="Fouriertransformation", speaker="Andreas", environment="A409"),
        Model("./data/TrainingData/A409/Fouriertransformation/Gustav/", word="Fouriertransformation", speaker="Gustav", environment="A409"),
        Model("./data/TrainingData/A409/Fouriertransformation/All/", word="Fouriertransformation", speaker="All", environment="A409"),

        Model("./data/TrainingData/G117/Zwei/Andreas/", word="Zwei", speaker="Andreas", environment="G117"),

        Model("./data/TrainingData/G117/Drei/Andreas/", word="Drei", speaker="Andreas", environment="G117"),

        Model("./data/TrainingData/G117/Fouriertransformation/Andreas/", word="Fouriertransformation", speaker="Andreas", environment="G117"),

        Model("./data/TrainingData/G117/Lisa/Andreas/", word="Lisa", speaker="Andreas", environment="G117"),
        Model("./data/TrainingData/G117/Lisa/Gustav/", word="Lisa", speaker="Gustav", environment="G117"),
        Model("./data/TrainingData/G117/Lisa/All/", word="Lisa", speaker="All", environment="G117"),

        Model("./data/TrainingData/G117/Gustav/Andreas/", word="Gustav", speaker="Andreas", environment="G117"),
        Model("./data/TrainingData/G117/Gustav/Gustav/", word="Gustav", speaker="Gustav", environment="G117"),
        Model("./data/TrainingData/G117/Gustav/All/", word="Gustav", speaker="All", environment="G117"),

        Model("./data/TrainingData/G117/Andreas/Andreas/", word="Andreas", speaker="Andreas", environment="G117"),
        Model("./data/TrainingData/G117/Andreas/Gustav/", word="Andreas", speaker="Gustav", environment="G117"),
        Model("./data/TrainingData/G117/Andreas/All/", word="Andreas", speaker="All", environment="G117"),

        Model("./data/TrainingData/G117/Moegen/Andreas/", word="Moegen", speaker="Andreas", environment="G117"),
        Model("./data/TrainingData/G117/Moegen/Gustav/", word="Moegen", speaker="Gustav", environment="G117"),
        Model("./data/TrainingData/G117/Moegen/All/", word="Moegen", speaker="All", environment="G117"),

        Model("./data/TrainingData/G117/Und/Andreas/", word="Und", speaker="Andreas", environment="G117"),
        Model("./data/TrainingData/G117/Und/Gustav/", word="Und", speaker="Gustav", environment="G117"),
        Model("./data/TrainingData/G117/Und/All/", word="Und", speaker="All", environment="G117"),

        Model("./data/TrainingData/G117/Signalverarbeitung/Andreas/", word="Signalverarbeitung", speaker="Andreas", environment="G117"),
        Model("./data/TrainingData/G117/Signalverarbeitung/Gustav/", word="Signalverarbeitung", speaker="Gustav", environment="G117"),
        Model("./data/TrainingData/G117/Signalverarbeitung/All/", word="Signalverarbeitung", speaker="All", environment="G117"),



    ])
    return modelList