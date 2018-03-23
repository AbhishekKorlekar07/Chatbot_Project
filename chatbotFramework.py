#things for nltk library
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import requests

#things we need for tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
import speech_recognition as sr
r = sr.Recognizer()

st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

#import chatbot intent file
import json

#restore all data structures
data = pickle.load(open("training_data","rb"))
words = data["words"]
classes = data["classes"]
train_x = data["train_x"]
train_y = data["train_y"]

#import our chatbot intents file
with open("intents.json") as json_data:
	intents = json.load(json_data)



#--------------------start-----------------------

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')



#-------------------end--------------------------

#load your saved model
model.load("./model.tflearn")


def clean_up_sentence(sentence):
	#tokenize the pattern
	sentence_words = nltk.word_tokenize(sentence)
	#stem each word
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	
	return sentence_words

#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
	#tokenize the pattern
	sentence_words = clean_up_sentence(sentence)
	print("sentence_words: ",sentence_words)

	#bag of words
	bag = [0] * len(words)
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s:
				bag[i] = 1
				if show_details:
					print("found in bag: %s" %w )

	return(np.array(bag))


def getWeatherInfo(sentence, tagName):
	tokenized_text = word_tokenize(sentence)
	result = st.tag(tokenized_text)
	print("result: ", result)
	locations = []
	locationsWeatherInfo = {"data" : {}, "locationFound" : False}

	for key, val in result:
		if val == "LOCATION":
			locations.append(key)

	if len(locations) != 0:
		for value in locations:
				weatherResultObject = requests.get("http://api.openweathermap.org/data/2.5/weather?q=" + value + "&units=imperial&APPID=4bd094be13717cc1a7f7e2ac3e4aa334")
				print("i want to see the status code: ",weatherResultObject.status_code)
				if weatherResultObject.status_code == 200:
					weatherResult = weatherResultObject.json()
					if tagName == "tempreture":
						locationsWeatherInfo["data"][value] = "The tempreture for " + value + " is " + str(weatherResult["main"]["temp"])
					else:
						locationsWeatherInfo["data"][value] = "The weather information for " + value + " is " + str(weatherResult["main"]["temp"]) + " degree fahrenheit, wind speed of " + str(weatherResult["wind"]["speed"]) + " with " + weatherResult["weather"][0]["description"] + " and humidity of " + str(weatherResult["main"]["humidity"])
						
					locationsWeatherInfo["locationFound"] = True

	return locationsWeatherInfo


context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
	# generate probabilities from the model
	results = model.predict([bow(sentence, words)])[0]
	print("results after model.predict: ",results)
	# filter out predictions below a threshold
	results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
	print("results after > than error threshold: ",results)
	# sort by strength of probability
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append((classes[r[0]], r[1]))
	# return tuple of intent and probability
	return return_list

def response(sentence, userID='123', show_details=False):
	results = classify(sentence)
	# if we have a classification then find the matching intent tag
	if results:
		# loop as long as there are matches to process
		while results:
			for i in intents['intents']:
				# find a tag matching the first result
				if i['tag'] == results[0][0]:
					print("i want to check what got captured: ", results)
					print("i want to check what got captured again: ", results[0][0])
					# set context for this intent if necessary
					if 'context_set' in i:
						if show_details: print ('context:', i['context_set'])
						context[userID] = i['context_set']

					# check if this intent is contextual and applies to this user's conversation
					if not 'context_filter' in i or \
						(userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
						if show_details: print ('tag:', i['tag'])
						if i['tag'] == "weather" or i['tag'] == "cityName" or i['tag'] == "tempreture":
							weatherResult = getWeatherInfo(sentence,i['tag'])
							weatherInformation = ""
							if weatherResult["locationFound"]:
								for key in weatherResult["data"]:
									weatherInformation = weatherInformation + weatherResult["data"][key]
								return weatherInformation
							else:
								return (random.choice(i['responses']))

						# a random response from the intent
						return (random.choice(i['responses']))

			results.pop(0)


while(True):
	x = raw_input("Enter your message: ")
	resp = response(x)
	print(resp)


# while(True):
#     with sr.Microphone() as source:
#        print("Please tell me: ")
#        audio = r.listen(source)

#     try:
#         print("Me: ",r.recognize_google(audio))
#         userText = r.recognize_google(audio)
#         x = userText
#         resp = response(x)
#         print(resp)
	
#     except:
#         pass