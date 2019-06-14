import numpy as np
import sys
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import os
import sys
import numpy as np
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import time
class TrainNewModel:
	def __init__(self, new_data, new_label):
		self.new_data=new_data
		self.new_label=new_label

	def train(self, classifier, X, y):
		start = time.time()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
		classifier.fit(X_train, y_train)
		end = time.time()
		print("Accuracy: " + str(classifier.score(X_test, y_test)) + ", Time duration: " + str(end - start))
		return classifier, classifier.score(X_test, y_test)

	def new_model(self):
		news = fetch_20newsgroups(subset='all')
		news_data1=news.data
		# for i in news.data:
		# 	news20_data.append(i)
		print(news_data1[0])
		print(self.new_data[0])
		news_data1.extend(self.new_data)
		news_label1 = np.append(news.target, self.new_label)
		#print(news_label[0])
		print(news_data1[0])
		#stemmer = SnowballStemmer("english")
		tfidfclassifier = Pipeline([ ('vectorizer', TfidfVectorizer( stop_words=stopwords.words('english') + list(string.punctuation))), ('classifier', LinearSVC(C=40,random_state=11))])
		X_train, X_test, y_train, y_test = train_test_split(news_data1, news_label1, test_size=0.2, random_state=11)
		newmodel1=tfidfclassifier.fit(X_train, y_train)
		acc = tfidfclassifier.score(X_test, y_test)
		#newmodel1,acc = train(tfidfclassifier, news_data, news_label)
		pickle.dump(newmodel1, open('model2.pkl','wb'))
		print("successfully dumped!")
		return acc


	




