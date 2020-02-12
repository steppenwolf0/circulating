import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
import pandas as pd 
def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./data/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./data/labels.csv", header=None)
		
	return dfData.as_matrix(), dfLabels.as_matrix().ravel() # to have it in the format that the classifiers like


plt.figure(figsize=(12, 12))


X, y = loadDataset()
numberOfFolds = 10
skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
indexes = [ (training, test) for training, test in skf.split(X, y) ]

labels=np.max(y)+1
yTest=[]
yNew=[]
cMatrix=np.zeros((labels, labels))
countCorrect=0
for train_index, test_index in indexes :
			
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	# let's normalize, anyway
	# MinMaxScaler StandardScaler Normalizer
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	kmeans = KMeans(n_clusters=10, max_iter=3000, algorithm = 'full').fit(X_train)
	y_new=kmeans.predict(X_test)
	
	yNew.append(y_new)
	yTest.append(y_test)
			
	for i in range(0,len(y_new)):
		cMatrix[y_test[i]][y_new[i]]+=1
		if (y_test[i]==y_new[i]):
			countCorrect=countCorrect+1

print(countCorrect/len(y))
#pd.DataFrame(cMatrix).to_csv("./data/cMatrix.csv", header=None, index =None)


#plt.subplot(111)
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.title("Unevenly Sized Blobs")

#plt.show()