#things for nltk library
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#things we need for tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

#import chatbot intent file
import json
with open("intents.json") as json_data:
	intents = json.load(json_data)	

words = []
classes = []
documents = []
ignore_words = ["?"]


for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		#tokenize each word in the sentence
		print(pattern)
		w = nltk.word_tokenize(pattern)
		print("w: ",w)
		#add to your word list
		words.extend(w)
		#add to documents in our corpus
		documents.append((w, intent["tag"]))
		#add classes to list
		if intent["tag"] not in classes:
			classes.append(intent["tag"])

print(words)
print("docs: ",documents)
print("classes: ",classes)

#stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

print("printing the words again: ",words)

#remove duplicates
classes = sorted(list(set(classes)))

print(len(documents), "documents",documents)
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words",words)


#create our training data
training = []
output = []
#create an empty array for our output
output_empty = [0] * len(classes)

print("output_empty: ",output_empty)

#training set, bag of words for each sentence
for doc in documents:	
	print("i want to see what doc is: ",doc)
	#initialize our bag of words
	bag = []
	#list of tokenized words for pattern
	pattern_words = doc[0]
	print("pattern_words first element: ", pattern_words)
	#stem each word
	pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
	print("pattern_words: ", pattern_words)
	#create our bag of words array
	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)


	#ouput is 0 for each tag and 1 for current tag
	output_row = list(output_empty)
	print("output_row 1st: ",output_row)
	output_row[classes.index(doc[1])] = 1
	print("output_row 2nd: ",output_row)

	training.append([bag, output_row])
	print("Training: ",training)


#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
print("after your using nparray on training: ", training)


#create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print("train_x: ", train_x)
print("train_y: ", train_y)


# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

#---------------------------start----------------

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

p = bow("what is the tempreture in dekalb?", words)
print (p)
print (classes)

print(model.predict([p]))


#----------------------------end-------------
#save all the data structures
pickle.dump({"words" : words, "classes" : classes, "train_x" : train_x, "train_y" : train_y}, open("training_data", "wb"))

