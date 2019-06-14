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

app = Flask(__name__)

def analytics(a,length):
    b=max(a)
    print(a)
    b=int(b)
    print(b)
    if(b>=20):
        new_array=np.zeros(b+1)
    else:
        new_array=np.zeros(20)
    print(len(new_array))
    for i in range(0, length):
        new_array[int(a[i])]= new_array[int(a[i])] + 1
    return new_array





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
	# my_data_file.write(content['color'])
	content=content.get("config")
	# print (content)
	process=[content]
	# print(process)
	for sub in process:
		for key in sub:
			for i in range(0,20):
				if(sub[key]==Category.category[i]):
					sub[key]=str(i)
	list_keys = [ k for k in process[0].keys() ]
	list_values = [k for k in process[0].values()]
	try:
		for sub in process:
			for key in sub:
				sub[key] = int(sub[key])
		print(process)
		length=len(list_values)
		for i in range(0,len(list_values)):
			list_values[i]=int(list_values[i])
		print(list_values)
		analytics_array=analytics(list_values,length)
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
	my_data_file = open('new_data.txt', 'w')
	my_data_file.write(str(list_keys))
	my_data_file = open('new_data_lables.txt', 'w')
	my_data_file.write(str(list_values))
	print(list_values)
	print(list_keys)
	newmodel= TrainNewModel(list_keys,list_values)
	accuracy=newmodel.new_model()
	return_string=return_string + "We have dumped new model having accuracy( "+ str(accuracy)+" )"
	return return_string  


if __name__ == '__main__':
	app.run(port=5000, debug=True)