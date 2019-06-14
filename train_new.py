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


	def new_model(self, old_data, old_label, stop_words, path):
		news_data1=old_data
		news_data1.extend(self.new_data)
		news_label1 = np.append(old_label, self.new_label)
		tfidfclassifier = Pipeline([ ('vectorizer', TfidfVectorizer(stop_words)), ('classifier', LinearSVC(C=40,random_state=11))])
		X_train, X_test, y_train, y_test = train_test_split(news_data1, news_label1, test_size=0.2, random_state=11)
		newmodel1=tfidfclassifier.fit(X_train, y_train)
		acc = tfidfclassifier.score(X_test, y_test)
		with open(path, 'wb') as file:
			pickle.dump(newmodel1, file)
		print("successfully dumped!")
		return acc


	




