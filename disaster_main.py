# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
print("program start")
import numpy as napi # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# Input data files are available in the "../input/" directory.
print("loading data")
sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

X = train.iloc[:, 1:-1]
y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X.head()

# keyword dictionary
keyword_col = X_train["keyword"]
keyword_not_null = keyword_col[keyword_col.notnull()]
keyword_arr = keyword_not_null.values
# keyword_arr = keyword_arr[keyword_arr.notnull()]
# print(keyword_arr.size)
keyword_arr = napi.unique(keyword_arr)
# print(keyword_arr.size)
KEYWORD_ARRAY_SIZE = keyword_arr.size

# TODO: Convert keywords to feature vectors
keyword_features = pd.get_dummies(X_train.iloc[:,0])
keyword_features.head()
# print("keyword_features shape: ", keyword_features.shape)
print("length of keyword feature vector", KEYWORD_ARRAY_SIZE)  # 221

'''
keyword_to_vector
Description: one-hot encode a keyword
Inputs: key -- keyword to convert (string)
         keyword_arr -- "dictionary" to be used (sorted array of strings)
                        Always use keyword_arr in this project
Returns: a feature vector (arrat of ints, 0 or 1)
'''
def keyword_to_vector(key, keyword_arr):
    feature_vect_bool = (keyword_arr == key)
    feature_vect = 1 * feature_vect_bool
    return feature_vect
    
# print(keyword_to_vector("ablaze", keyword_arr))

# location dictionary
location_arr = X_train["location"][X_train["location"].notnull()].values
# location_arr = location_arr[location_arr.notnull()]
print(location_arr.size)
location_arr = napi.unique(location_arr)
print(location_arr.size)
LOCATION_ARRAY_SIZE = location_arr.size
# print(location_arr)

# TODO: Build a dictionary on text
sentences = X_train['text'].tolist()
cv = CountVectorizer(min_df=0.0008)
X = cv.fit_transform(sentences).toarray(float)
# print(len(sentences))
# print(cv.vocabulary_)
# print(X[100])
print("length of text feature vector: ", len(cv.get_feature_names()))
