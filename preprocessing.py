import csv
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

random.seed(0)

#STOPWORDS = set(stopwords.words('english'))
VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
MAX_LENGTH = 25
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = '<OOV>'
TRIAINING_PORTION = .8


heading= []
labels = []



def data_loading(heading, labels, path, label_value):
	"""
	heading: list where data is to be appended
	labels: list where labels can be appended
	path: path to the file
	label_value: if clickbait then 1, else 0

	"""

	for lines in open(path, encoding="utf8").readlines():
		if(lines != "\n"):
			heading.append(lines.split("\n")[0])
			labels.append(label_value)

	return heading, labels



def train_val_split(headings, labels, train_fraction = 0.8):
	"""
	headings: full set of heading data
	labels: full set of labels data
	train_fraction: fraction for training set

	"""
	# to shuffle the lists before split
	temp = list(zip(headings, labels))
	random.shuffle(temp)
	list1, list2 = zip(*temp)


	len_train = int(len(list1) * train_fraction)

	train_headings = list1[0:len_train]
	train_labels = list2[0:len_train]

	val_headings = list1[len_train:]
	val_labels = list2[len_train:]

	return train_headings, train_labels, val_headings, val_labels


def tokenizer(sequence_list):
	tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token = OOV_TOK)
	tokenizer.fit_on_texts(sequence_list)
	#word_index = tokenizer.word_index  #list of all tokens created
	
	return tokenizer



def apply_tokenizer(tokenizer, sequence_list):
	train_sequence = tokenizer.texts_to_sequences(sequence_list)
	train_padded = pad_sequences(train_sequence, maxlen = MAX_LENGTH, padding = PADDING_TYPE, truncating = TRUNC_TYPE)

	return train_padded



def LSTM_model(num_epochs, train_padded, train_label, val_padded, val_labels):
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_DIM)),
		tf.keras.layers.GaussianNoise(0.5),
		tf.keras.layers.Dense(EMBEDDING_DIM, activation = 'relu'),
		tf.keras.layers.Dense(2, activation= 'sigmoid')
		])

	model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	earlystopping = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor = 'val_loss', mode = 'min', save_best_only = True)
	history = model.fit(train_padded, train_label, epochs = num_epochs, validation_data = (val_padded, val_labels), verbose = 2, callbacks = [earlystopping])

	return model.summary(), history


def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history['val_'+ string])
	plt.xlabel("Epochs")
	plt.legend([string, 'val_' + string])
	plt.show()



def text_for_pred(tokenizer, txt, model):
	padded = apply_tokenizer(tokenizer, txt)
	pred = model.predict(padded)
	return pred 







data_X, label_Y = data_loading(heading=heading, labels = labels, path = "data/clickbait_data", label_value = 1)
print("*****************************", len(data_X))
data_X, label_Y = data_loading(heading=data_X, labels = label_Y, path = "data/non_clickbait_data", label_value = 0)
print("*****************************", len(data_X))



train_headings, train_labels, val_headings, val_labels = train_val_split(data_X, label_Y)
train_labels= np.array(train_labels)
val_labels= np.array(val_labels)

tokenizer = tokenizer(train_headings)
train_headings_padded = apply_tokenizer(tokenizer, train_headings)
val_headings_padded = apply_tokenizer(tokenizer, val_headings)

summary, history = LSTM_model(100, train_headings_padded, train_labels, val_headings_padded, val_labels)

#plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')