"""

`     https://www.kaggle.com/ronitf/heart-disease-uci


"""
import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("heart.csv")

data.describe()

data.info()

data.dtypes

x = data.drop(["target"],axis =1)

y = data["target"]

sns.countplot(data["slope"],hue = data["target"])

sns.countplot(data["target"])

#sns.pairplot(data,hue = "target")


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)



from sklearn.preprocessing import StandardScaler as ss

ss = ss()

x_train = ss.fit_transform(x_train)

x_test  = ss.transform(x_test)


from sklearn.decomposition import PCA

pc = PCA( n_components=7)

x_train = pc.fit_transform(x_train)

x_test = pc.transform(x_test)

pca = pc.explained_variance_ratio_



from sklearn.tree import DecisionTreeClassifier as dt

dt = dt()

dt.fit(x_train,y_train)


dt.score(x_train,y_train)


y_pr = dt.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix (y_test,y_pr)

cr = classification_report (y_test,y_pr)


from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(n_estimators=168)

rf.fit(x_train,y_train)


rf.score(x_train,y_train)

y_pr = rf.predict(x_test)


from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix (y_test,y_pr)

cr = classification_report (y_test,y_pr)

sns.heatmap(cm,annot = True,cbar =False)


"""
from sklearn.model_selection import GridSearchCV

para = {"n_estimators": [165,160,169,168,170,173,175,180],"criterion" : ["gini", "entropy"]}

gs  = GridSearchCV(estimator = rf ,param_grid = para ,  cv= 10)

gs.fit(x_train,y_train)

gs.best_params_

y_pr = gs.predict(x_test)


from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix (y_test,y_pr)

cr = classification_report (y_test,y_pr)
"""