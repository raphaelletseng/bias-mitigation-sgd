import sys
sys.path.append('C:\\Users\\rapha\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\fairlearn')

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = fetch_openml(data_id=1590, as_frame=True)
X = pd.get_dummies(data.data)
print(X)
y_true = (data.target == '>50K') * 1
sex = data.data['sex']
sex.value_counts()
print(len(X))

classifier = DecisionTreeClassifier(min_samples_leaf = 10, max_depth = 4)
classifier.fit(X,y_true)
y_pred = classifier.predict(X)
gm = MetricFrame(accuracy_score, y_true, y_pred, sensitive_features = sex)
print(gm.overall)
print('#-----------------------------------#')
print(gm.by_group)

sr = MetricFrame(selection_rate, y_true, y_pred, sensitive_features = sex)
sr.overall
sr.by_group
