import os
import numpy as np
from tkinter import *
from tkinter import ttk, font
from tkinter.filedialog import askopenfilename
from AppFunctions import loadModels
from HMM.Viterbi import concatModels, performViterbiForPrediction, trackBackwardPointerForPrediction
from AppStyling import styleElements
# Setup Print Options
np.set_printoptions(precision=2, suppress=True)

class App(object):
    def __init__(self, master):
        self.master = master
        self.labelFont = font.Font(family="Helvetica", size=16, weight="bold")
        self.smallLabelFont = font.Font(family="Helvetica", size=11, weight="bold")
        self.master.geometry("1390x785")
        self.master.title("Speech Recognition with HMMs")
        self.master.configure(background='white')
        self.master.bind('<Escape>', lambda e: root.quit())
        self.selectedSilenceModel = IntVar()
        self.selectedSilenceModel.set(1)
        self.createWidgets()
        self.placeWidgets()
        styleElements()

    def browse(self):
        browsedFilePath = askopenfilename(initialdir=os.getcwd()+"/data/")
        browsedFileName = os.path.basename(browsedFilePath)
        self.treeViewReferences.insert('', 'end', text=len(self.treeViewReferences.get_children()), values=(browsedFileName, browsedFilePath))

    def loadModels(self):
        self.modelList = loadModels()
        for i, item in enumerate(self.modelList):
            self.treeViewModels.insert('', 'end', text=i, values=(item.word, item.speaker, item.environment))

    def commitModels(self):
        focusedItem = self.treeViewModels.focus()
        itemId = self.treeViewModels.item(focusedItem)['text']
        a = self.modelList[itemId].word
        b = self.modelList[itemId].speaker
        c = self.modelList[itemId].environment
        self.treeViewCommitedModels.insert('', 'end', text=itemId, values=(a, b, c))

    def deleteItemFromCommitedModels(self, event):
        focusedItem = self.treeViewCommitedModels.focus()
        itemId = self.treeViewModels.item(focusedItem)['text']
        self.treeViewCommitedModels.delete(focusedItem)

    def addItemToCommitedModels(self, event):
        focusedItem = self.treeViewModels.focus()
        itemId = self.treeViewModels.item(focusedItem)['text']
        a = self.modelList[itemId].word
        b = self.modelList[itemId].speaker
        c = self.modelList[itemId].environment
        self.treeViewCommitedModels.insert('', 'end', text=itemId,
                                   values=(a, b, c))

    def printModelInfoForSelected(self):
        focusedItem = self.treeViewModels.focus()
        itemId = self.treeViewModels.item(focusedItem)['text']
        modelInfoText = self.modelList[itemId].printModelInformation()
        self.modelInfoBox.delete(1.0, END)
        self.modelInfoBox.insert(END, modelInfoText)
        self.modelInfoBox.update()

    def trainSelected(self):
        focusedItem = self.treeViewModels.focus()
        itemId = self.treeViewModels.item(focusedItem)['text']
        self.modelList[itemId].train(self.modelInfoBox)

    def initTreeViewForModels(self):
        self.treeViewModels.heading('#0', text='#')
        self.treeViewModels.heading('#1', text='Word')
        self.treeViewModels.heading('#2', text='Speaker')
        self.treeViewModels.heading('#3', text='Room')
        self.treeViewModels.column('#0', width=50, anchor=CENTER, stretch=False)
        self.treeViewModels.column('#1', width=150, anchor=CENTER, stretch=False)
        self.treeViewModels.column('#2', width=150, anchor=CENTER, stretch=False)
        self.treeViewModels.column('#3', width=50, anchor=CENTER, stretch=False)
        self.treeViewModels.bind("<Double-1>", self.addItemToCommitedModels)

    def initTreeViewForCommitedModels(self):
        self.treeViewCommitedModels.heading('#0', text='#')
        self.treeViewCommitedModels.heading('#1', text='Word')
        self.treeViewCommitedModels.heading('#2', text='Speaker')
        self.treeViewCommitedModels.heading('#3', text='Room')
        self.treeViewCommitedModels.column('#0', width=50, anchor=CENTER, stretch=False)
        self.treeViewCommitedModels.column('#1', width=150, anchor=CENTER, stretch=False)
        self.treeViewCommitedModels.column('#2', width=150, anchor=CENTER, stretch=False)
        self.treeViewCommitedModels.column('#3', width=50, anchor=CENTER, stretch=False)
        self.treeViewCommitedModels.bind("<Double-1>", self.deleteItemFromCommitedModels)

    def initTreeViewForResult(self):
        self.treeViewResult.heading('#0', text='#')
        self.treeViewResult.heading('#1', text='Word')
        self.treeViewResult.heading('#2', text='Speaker')
        self.treeViewResult.heading('#3', text='Room')
        self.treeViewResult.heading('#4', text='Cost')
        self.treeViewResult.column('#0', width=50, anchor=CENTER, stretch=False)
        self.treeViewResult.column('#1', width=150, anchor=CENTER, stretch=False)
        self.treeViewResult.column('#2', width=150, anchor=CENTER, stretch=False)
        self.treeViewResult.column('#3', width=100, anchor=CENTER, stretch=False)
        self.treeViewResult.column('#4', anchor=CENTER, stretch=True)

    def initTreeViewForReferences(self):
        self.treeViewReferences.heading('#0', text='#')
        self.treeViewReferences.heading('#1', text='Word or Sequence of Words')
        self.treeViewReferences.heading('#2', text='')
        self.treeViewReferences.heading('#3', text='')
        self.treeViewReferences.column('#0', width=50, anchor=CENTER, stretch=False)
        self.treeViewReferences.column('#1', width=500, anchor=W, stretch=False)
        self.treeViewReferences.column('#2', width=0, anchor=W, stretch=False)
        self.treeViewReferences.column('#3', width=0, anchor=W, stretch=False)

    def modelClicked(self, event):
        item = self.treeViewModels.focus()

    def predict(self):
        if self.selectedSilenceModel.get() == 1: # A409 Silence Model
            selectedSilenceModel = self.modelList[0]
        else:
            selectedSilenceModel = self.modelList[1]

        modelsToConcat = []
        for child in self.treeViewCommitedModels.get_children():
            itemId = self.treeViewCommitedModels.item(child)['text']
            model = self.modelList[itemId]
            modelsToConcat.append(model)

        # Create One Big Model
        bigMean, bigVariance, bigTransProbs, \
        modelIndexStart, modelIndexEnd, \
        wordArray, speakerArray, environmentArray = concatModels(selectedSilenceModel, modelsToConcat)

        # Selected Reference to predict
        focusedItem = self.treeViewReferences.focus()
        itemName = self.treeViewReferences.item(focusedItem)['values'][0]
        itemPath = self.treeViewReferences.item(focusedItem)['values'][1]
        reference = np.genfromtxt(itemPath, delimiter=' ')

        # traverse Trellis diagram
        backwardPointers, costMatrix = performViterbiForPrediction(reference, bigMean, bigVariance, bigTransProbs,
                                                               modelIndexStart, modelIndexEnd)

        # track Backward Pointer
        outputWord, outputSpeaker, outputCost, outputEnvironemnt = trackBackwardPointerForPrediction(reference, backwardPointers, wordArray, speakerArray, environmentArray, modelIndexEnd, costMatrix)

        # Print Result
        for i in range(len(outputWord)):
            self.treeViewResult.insert('', 'end', text=i+1, values=(outputWord[i], outputSpeaker[i], outputEnvironemnt[i], outputCost[i]))
        self.treeViewResult.insert('', 'end', text="--", values=("-------", "-------", "-------", "-------"))
        self.treeViewResult.update()
        self.treeViewResult.yview_moveto(1)

    def loadInitialReferences(self):
        for root, dirs, files in os.walk(os.getcwd() + "/data/ReferencesPresentation"):
            for name in files:
                self.treeViewReferences.insert('', 'end', text=len(self.treeViewReferences.get_children()), values=(name, root + "/" + name))

    def createWidgets(self):

        self.modelsLabel = Label(self.master, text='All Available Models', bg="#18384e", foreground="white", font=self.labelFont)
        self.treeViewModels = ttk.Treeview(self.master, columns=('Word', 'Speaker', 'Room'), height=20, style="Custom.Treeview")
        self.infoLabel = Label(self.master, text='Information', width=25, bg="#f47f20", foreground="white", font=self.labelFont)
        self.resultLabel = Label(self.master, text='Result', width=25, bg="#70be44", foreground="white", font=self.labelFont)
        self.initTreeViewForModels()
        self.getModelInfoButton = Button(self.master, text='Get Model Info', bg="#dddddd", foreground="#18384e", command=self.printModelInfoForSelected)
        self.modelInfoBox = Text(self.master, height=13, bg="#dddddd", relief=SOLID, state=NORMAL)
        self.loadModels()
        self.trainSelectedButton = Button(self.master, text='Train Selected', bg="#18384e", foreground="white", command=self.trainSelected)
        self.silenceModelLabel = Label(self.master, text='Select Silence Model:', bg="white", foreground="#18384e", font=self.smallLabelFont)
        self.commitedModelsLabel = Label(self.master, text='Commited Models', bg="#18384e", foreground="white", font=self.labelFont)
        ttk.Style().configure('TRadiobutton', background="white", foreground="#18384e")
        self.rb1 = ttk.Radiobutton(self.master, text='A409', variable=self.selectedSilenceModel, value=1)
        self.treeViewCommitedModels = ttk.Treeview(self.master, columns=('Word', 'Speaker', 'Room'), height=20, style="Custom.Treeview")
        self.rb2 = ttk.Radiobutton(self.master, text='G117', variable=self.selectedSilenceModel, value=2)
        self.initTreeViewForCommitedModels()
        self.referencesLabel = Label(self.master, text='References', bg="#18384e", foreground="white", font=self.labelFont)
        self.treeViewReferences = ttk.Treeview(self.master, columns=('Wort oder Wortfolge'), height=20, style="Custom.Treeview")
        self.loadInitialReferences()
        self.browseButton = Button(self.master, text="Browse", bg="#dddddd", foreground="#18384e", command=self.browse)
        self.predictButton = Button(self.master, text="Predict", bg="#18384e", foreground="white", command=self.predict)
        self.treeViewResult = ttk.Treeview(self.master, columns=('Wort', 'Speaker', 'Room', 'Cost'), height=10, style="Custom.Treeview")
        self.initTreeViewForReferences()
        self.initTreeViewForResult()

    def placeWidgets(self):
        self.modelsLabel.grid(column=0, row=0, columnspan=2, sticky="we", padx=5, pady=5)
        self.treeViewModels.grid(column=0, row=1, columnspan=2,  sticky="we", padx=5, pady=5)
        self.trainSelectedButton.grid(column=0, row=2, sticky="we", padx=5, pady=5)
        self.getModelInfoButton.grid(column=1, row=2, sticky="we", padx=5, pady=5)
        self.silenceModelLabel.grid(column=2, row=2, sticky="we", padx=5, pady=5)
        self.commitedModelsLabel.grid(column=2, row=0, sticky="we", padx=5, pady=5, columnspan=3)
        self.rb1.grid(column=3, row=2, sticky=W, padx=0, pady=5)
        self.rb2.grid(column=4, row=2, sticky=W, padx=5, pady=5)
        self.treeViewCommitedModels.grid(column=2, row=1, sticky="we", padx=5, pady=5, columnspan=3)
        self.referencesLabel.grid(column=5, row=0, sticky="ew", padx=5, pady=5, columnspan=2)
        self.treeViewReferences.grid(column=5, row=1, sticky="ew", padx=5, pady=5, columnspan=2)
        self.browseButton.grid(column=5, row=2, sticky="ew", padx=5, pady=5)
        self.predictButton.grid(column=6, row=2, sticky="ew", padx=5, pady=5)
        self.infoLabel.grid(column=0, row=3, sticky="we", padx=5, pady=5, columnspan=3)
        self.resultLabel.grid(column=3, row=3, sticky="we", padx=5, pady=5, columnspan=5)
        self.modelInfoBox.grid(column=0, row=4, sticky="nwe", padx=5, pady=5, columnspan=3)
        self.treeViewResult.grid(column=3, row=4, sticky="we", padx=5, pady=5, columnspan=5)

root = Tk()
app = App(root)
root.mainloop()
