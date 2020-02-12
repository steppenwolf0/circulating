# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from scipy import stats
# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# this is an incredibly useful function
from pandas import read_csv

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./data/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./data/labels.csv", header=None)
		
	return dfData.as_matrix(), dfLabels.as_matrix().ravel() # to have it in the format that the classifiers like


def runFeatureReduce() :
	
	orig_stdout = sys.stdout
	f = open('./data/manova.txt', 'w')
	sys.stdout = f
	
	print("Loading dataset...")
	X, y = loadDataset()
	
	maov = MANOVA(X,y)
	
	
	print(len(X))
	print(len(X[0]))
	print(len(y))

	print(maov.mv_test())
	
	est = sm.OLS(y, X)
	est2 = est.fit()
	print(est2.summary())
	sys.stdout = orig_stdout
	f.close()
	return

if __name__ == "__main__" :
	sys.exit( runFeatureReduce() )