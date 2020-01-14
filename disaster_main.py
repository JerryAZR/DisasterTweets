# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
print("program start")
import numpy as napi # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer, LancasterStemmer
# Input data files are available in the "../input/" directory.
print("loading data")
sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

X = train.iloc[:, 1:-1]
y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X.head()

'''
# keyword vectorizer
# Could instead use a CountVectorizer
keyword_col = X_train["keyword"]
keyword_not_null = keyword_col[keyword_col.notnull()]
keyword_arr = keyword_not_null.values
# keyword_arr = keyword_arr[keyword_arr.notnull()]
# print(keyword_arr.size)
keyword_arr = napi.unique(keyword_arr)
# print(keyword_arr.size)
KEYWORD_ARRAY_SIZE = keyword_arr.size

# TODO: Convert keywords to feature vectors
keyword_features = pd.get_dummies(X_train.iloc[:,0]).values
# print("keyword_features shape: ", keyword_features.shape)
print("length of keyword feature vector", KEYWORD_ARRAY_SIZE)  # 221
'''

'''
keyword_to_vector
Description: one-hot encode a keyword
Inputs: key -- keyword to convert (string)
         keyword_arr -- "dictionary" to be used (sorted array of strings)
                        Always use keyword_arr in this project
Returns: a feature vector (arrat of ints, 0 or 1)
'''
'''
def keyword_to_vector(key, keyword_arr):
    if pd.isna(key):
        return napi.zeros(len(keyword_arr))
    feature_vect_bool = (keyword_arr == key)
    feature_vect = 1 * feature_vect_bool
    return feature_vect

# CountVectorizer Approach:
# keywords = X_train['text'].tolist()
# keyword_cv = CountVectorizer()
# keyword_vect = keyword_cv.fit_transform(keywords).toarray(float)    
# print(keyword_to_vector("ablaze", keyword_arr))
'''

# keyword to numerical
# NOTE: Try to decrease number of features associated with keywords
keyword_dict = {}
# Right now we are using the entire training set (including train, cv, test) to do this;
# This is NOT good practice, but we may leave it because we are lazy.
keyword_full_arr = keyword_col = train[["keyword","target"]].values
stemmer = PorterStemmer() # could also try other stemmers
# print(keyword_full_arr)
for pair in keyword_full_arr:
    if pd.isna(pair[0]):
        stemmed_key = napi.nan
    else:
        stemmed_key = stemmer.stem(pair[0])
    
    if stemmed_key not in keyword_dict:
        keyword_dict[stemmed_key] = [0,0]
    keyword_dict[stemmed_key][0] += pair[1]
    keyword_dict[stemmed_key][1] += 1

# Next, transform keywords to numerical values:
numerical_keyword_features = []
for keyword in X_train["keyword"].values:
    if pd.isna(keyword):
        key = napi.nan
    else:
        key = stemmer.stem(keyword)
    numerical_keyword_features.append(keyword_dict[key][0]/keyword_dict[key][1])

numerical_keyword_features = (napi.array(numerical_keyword_features)).reshape(-1,1)

# Skip this part now (ignore location)
# # location dictionary
# location_arr = X_train["location"][X_train["location"].notnull()].values
# # location_arr = location_arr[location_arr.notnull()]
# print(location_arr.size)
# location_arr = napi.unique(location_arr)
# print(location_arr.size)
# LOCATION_ARRAY_SIZE = location_arr.size
# # print(location_arr)

# TODO: Build a dictionary on text
sentences = X_train['text'].tolist()
cv = CountVectorizer(min_df=0.0008)
text_features = cv.fit_transform(sentences).toarray(float)
# print(len(sentences))
# print(cv.vocabulary_)
# print(text_features[100])
print("length of text feature vector: ", len(cv.get_feature_names()))

# merge all the features:
print(keyword_features.shape, text_features.shape)
All_features = napi.concatenate((numerical_keyword_features, text_features), axis=1)
print(All_features.shape)

# stopword count
# stopword_count = []
# for tweet in sentences:
#     word_count.append(len(tweet.split()))

# Apply PCA if necessary
print("Doing PCA...")
pca = PCA(0.95)
pca.fit(All_features)
PCA(copy=True, iterated_power='auto', random_state=42, svd_solver='auto', tol=0.0)

train_X_pca = pca.transform(All_features)
print("n components: ", pca.n_components_)
# plt.figure()
# plt.plot(napi.cumsum(pca.explained_variance_ratio_))
# plt.title('Explained Variance')
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

# svm_C = 1.0
'''
Without PCA:
{'fit_time': array([48.45315099, 49.18679333, 49.14396572]), 
'score_time': array([22.64499092, 23.42073154, 23.34550667]), 
'test_score': array([0.64055591, 0.68082524, 0.67186563]), 
'train_score': array([0.70620605, 0.69862595, 0.71303824])}
'''

'''
with PCA
using numerical_keyword_features
svm_C = 1
svm_kernel = "rbf"
svm_gamma = "scale" # "auto" or "scale"

'fit_time': array([21.40067339, 21.66516542, 21.43422079]), 
'score_time': array([ 9.91457796, 10.0454669 ,  9.95311713]), 
'test_score': array([0.72622478, 0.75140607, 0.72983644]), 
'train_score': array([0.7757987 , 0.77254902, 0.77514124])
'''

svm_C = 1
svm_kernel = "rbf"
svm_gamma = "scale" # "auto" or "scale"
svm_1 = SVC(C=svm_C, kernel=svm_kernel, gamma=svm_gamma)
score_1 = cross_validate(svm_1, train_X_pca, y_train, cv=3, scoring="f1")
print(score_1)
