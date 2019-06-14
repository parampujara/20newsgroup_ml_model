# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import string
from category import Category
import operator
from train_new import TrainNewModel
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import analytics
import dataman

app = Flask(__name__)




model = pickle.load(open('model.pkl','rb'))

@app.route('/predict',methods=['POST'])

def predict():
	data = request.get_json(force=True)
	prediction = model.predict([data['stri']])
	output = int(prediction[0])
	output = Category.category[output]
	return jsonify(output)

  
@app.route('/postjson', methods = ['POST'])

def postJsonHandler():
	print (request.is_json)
	content = request.get_json()
	content=content.get("config")
	process=[content]
	cates=Category.category
	arr= list(range(0,20))
	dictionary=dict(zip(cates,arr))
	list1=dataman.datamanuplation(process,dictionary)
	if(list1== None):
		return "You have got unvalid labels!"
	list_keys = [ k for k in process[0].keys() ]
	list_values = list1
	
	try:
		length=len(list_values)
		list_values= [int(k) for k in list_values]
		analytics_array=analytics.analytics(list_values,length)
		return_string = "\n"
		temp=0
		news_num = list(map(operator.add, analytics_array,Category.news_num))
		for i in analytics_array:
			if(i != 0):
				return_string=return_string+"class:  " +str(temp) + "\t values:  " + str(i) + "\n"
			temp = temp + 1
		print(news_num)
		return_string= return_string + "Okay! We have got your data! \n"
	except ValueError:
		print("Input is not valid...")
		return_string="Oops! Not Valid Input file!"
	print(list_values)
	print(list_keys)
	news = fetch_20newsgroups(subset='all')
	newmodel= TrainNewModel(list_keys,list_values)
	stop_words=stopwords.words('english') + list(string.punctuation)
	path = "model2.pkl"
	accuracy=newmodel.new_model(news.data, news.target, stop_words, path)
	return_string=return_string + "We have dumped new model having accuracy( "+ str(accuracy)+" )"
	return return_string  


if __name__ == '__main__':
	app.run(port=5000, debug=True)