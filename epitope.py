#Classify B-cell linear epitopes from non-epitopes using amino acid count and dipeptide counts.
#Datasets obtained from Immune epitope database(IEDB)

import collections
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

negative = "B-negative.txt"
positive = "B-positive.txt"


AA = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
diAA = list(itertools.product(AA, repeat=2))
diAA = [a[0]+a[1] for a in diAA]

X=[]
y=[]

for trainingFile in [negative, positive]:
	f = open (trainingFile,"r")	
	for line in f:
		aadict=collections.OrderedDict()
		diaadict = collections.OrderedDict()
		for aa in AA:
			aadict[aa]=0
		for diaa in diAA:
			diaadict[diaa]=0
		if ">" not in line:
			for aa in list(line.strip()):
				aadict[aa]+=1

			length = sum(aadict.values())
			aadict.update((x, y/float(length)) for x, y in aadict.items())
			aac = aadict.values()

			for i in range(len(line)-2):
				dipep = line[i]+line[i+1]
				diaadict[dipep]+=1

			diaadict.update((x, y/float(length)) for x, y in diaadict.items())
			dpc = diaadict.values()
			aac=aac+dpc

			if trainingFile == negative:
				y.append(0)
			else:
				y.append(1)
			X.append(aac)


X = pd.DataFrame(X, columns=AA+diAA)
#X = pd.DataFrame(X, columns=AA)
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3,random_state=109)

clf = svm.SVC(kernel='rbf', C=0.1, gamma=0.01, probability = True) 

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

lr_probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print ("ROC",lr_auc)


