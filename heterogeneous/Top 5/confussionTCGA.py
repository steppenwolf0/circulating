import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

temp=read_csv("./data/cMatrix0.csv", header=None, sep=',')
s0=len(temp)	
s1=len(temp[0]) # classes
resultMatrix=np.zeros((8,10))
for i in range(0,8):
	Data = (read_csv("./data/cMatrix"+str(i)+".csv", header=None, sep=',')).values
	
	sumVal=[]
	for j in range(0,s0):
		sum=0
		for k in range(0,s1):
			sum=sum+Data[j][k]
		sumVal.append(sum)
	
	for j in range(0,s0):
		resultMatrix[i][j]=Data[j][j]/sumVal[j]
		


# Display matrix
dfClasses = read_csv("./data/classes.csv", header=None)	
pd.DataFrame(resultMatrix).to_csv("./data/resultMatrix.csv", header=None, index =None)
dfData=(resultMatrix)

ylabels=["Gradient Boosting","Random Forest", "Logistic Regression", "Passive Aggressive",
"SGD", "SVC", "Ridge", "Bagging"]	
xlabels = dfClasses.values.ravel()
f, ax = plt.subplots(figsize=(20,5))
#ax = sn.heatmap(dfData, annot=True, fmt="d", cmap=sn.color_palette("Blues"),  cbar=False)
#palette = sn.color_palette("GnBu_d",30)
palette = sn.color_palette("Blues",30)
#palette.reverse()
sn.set(font_scale = 1.5)
ax = sn.heatmap(dfData, annot=True, annot_kws={"size": 12}, fmt=".4f", cmap=palette,    
#cbar_kws={'label': 'Accuracy 'orientation': 'horizontal'},
cbar_kws={'label': 'Accuracy' },
cbar=True, xticklabels=xlabels, yticklabels=ylabels)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Cancer Type', fontsize=12)
ax.set_ylabel("Classifier", fontsize=12)
plt.show()
plt.savefig("test")



