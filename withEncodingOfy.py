
''' predicting the type of article by analysing the topics '''
# trying to categorize topics of the articles 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv('News_topics.csv')

# cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
y = dataset.iloc[:, 1].values
# y is a categorical data, it needs to be encoded 
from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
y = le_y.fit_transform(y)


corpus = []
for i in range(0,10):
    review = re.sub('[^a-zA-Z]',' ', dataset['Topics'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

# constructing a bag of words to study the meanings:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
#y = dataset.iloc[:, 1].values


#classification using Naive Bayes 

# splittting the dataset into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

'''# Feature Scalling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # not required for binary dependent variable'''

# fitting into NaiveBayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predicting the test set
y_pred = classifier.predict(X_test)

# building confusion matrix for the ratio of guesses 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




    

    
    
