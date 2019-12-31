# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as napi # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# Input data files are available in the "../input/" directory.

sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

X = train.iloc[:, 1:-1]
y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X.head()

# keyword dictionary
keyword_arr = X_train.iloc[:,1].values
keyword_arr = keyword_arr[~pd.isnull(keyword_arr)]
print(keyword_arr.size)
keyword_arr = napi.unique(keyword_arr)
print(keyword_arr.size)
KEYWORD_ARRAY_SIZE = keyword_arr.size

# TODO: Coonvert keywords to feature vectors

# location dictionary
location_arr = X_train.iloc[:,2].values
location_arr = location_arr[~pd.isnull(location_arr)]
print(location_arr.size)
location_arr = napi.unique(location_arr)
print(location_arr.size)
LOCATION_ARRAY_SIZE = location_arr.size
print(location_arr)

# TODO: Build a dictionary on text
sentences = X_train['text'].tolist()
cv = CountVectorizer(min_df=0.01)
X = cv.fit_transform(sentences).toarray(float)
print(cv.vocabulary_)
print(X[100])
