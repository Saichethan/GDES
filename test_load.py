import os
import re
import pickle
import csv
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


data_dir = "test/"

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ' ')    
    return text


def load_test():

	size = 300
	sentences  = []  # list of sentences
	position = [] # sentence position score
	rplen = [] # list of len of RP's
	plen = []
	clen = 0
	full_text = []
	plen.append(0)
	lsentences = []

	for item in range(1000, 1022):
		data = open(data_dir + str(item) + ".txt", 'r')
		data_list = data.readlines()
		clen = clen + len(data_list)
		plen.append(clen)
		for i in range(len(data_list)):
			sl = strip_links(data_list[i].lower())
			lsentences.append(sl)
			sentences.append(data_list[i])
			position.append(float(1/(1+i)))
		
	position = np.asarray(position)

	#print("Max Sentence Length: ",maxlen)
	tokenizer = Tokenizer(oov_token="<OOV>") #oov_token="<OOV>"
	tokenizer.fit_on_texts(lsentences)

	sequences = tokenizer.texts_to_sequences(lsentences)

	padding = pad_sequences(sequences,padding="post",truncating="post",maxlen=200)
	
	X_test = np.asarray(padding)
	P_test = np.asarray(position)


	sent_test = sentences[:]

	
	los = np.asarray(plen)
 

	return X_test, P_test, sent_test, los
