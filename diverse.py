import os
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import operator
import gensim

def cleanData(sentence):
	sentence = re.sub('[^A-Za-z0-9 ]+', '', sentence)
	#sentence filter(None, re.split("[.!?", setence))
	ret = []
	#sentence = stemmer.stem(sentence)	
	for word in sentence.split():
		#if not word in stopwords:
		ret.append(word)
	return " ".join(ret)


def getVectorSpace(cleanSet):
	vocab = {}
	for data in cleanSet:
		for word in data.split():
			vocab[data] = 0
	return vocab.keys()
	

def calculateSimilarity(sentence, doc):

	if doc == []:
		return 0
	vocab = {}
	for word in sentence:
		vocab[word] = 0
	
	docInOneSentence = '';
	for t in doc:
		docInOneSentence += (t + ' ')
		for word in t.split():
			vocab[word]=0	

	docum = ""
	for item in doc:
		docum = docum + " " + item
	dtotal = doc[:]
	dtotal.append(sentence)


	vectorizer = TfidfVectorizer()
	vectorizer.fit(dtotal)

	vector1 = vectorizer.transform([sentence])		
	temp1 = vector1.toarray()
	temp1 = temp1.tolist()

	vector2 = vectorizer.transform([docum])		
	temp2 = vector2.toarray()
	temp2 = temp2.tolist()

	
	return cosine_similarity(temp2, temp1)[0][0]
	
"""
#import gensim.downloader as api
#WMD = api.load('word2vec-google-news-300')

WMD = gensim.models.KeyedVectors.load_word2vec_format('assets/wiki-news-300d-1M.vec')
print("Loaded")

def calculateSimilarity(sentence, doc):

	if doc == []:
		return 0
	vocab = {}
	for word in sentence:
		vocab[word] = 0
	
	docInOneSentence = '';
	for t in doc:
		docInOneSentence += (t + ' ')
		for word in t.split():
			vocab[word]=0	

	docum = ""
	for item in doc:
		docum = docum + " " + item

	s1 = str(sentence).lower().split()
	s2 = str(docum).lower().split()
	#print(s1,s2)

	s1 = [w for w in s1]
	s2 = [w for w in s2]

	distance = WMD.wmdistance(s1,s2)

	return 1/(1+distance)

"""


def mmr(texts, pred):

	sentences = []
	clean = []
	originalSentenceOf = {}
	pos = {}
		
	for ind in range(len(texts)):
		cl = texts[ind] #cleanData(texts[ind])
		sentences.append(texts[ind])
		clean.append(cl)
		originalSentenceOf[cl] = texts[ind]
		pos[cl] = ind		
	setClean = set(clean)
		
	#calculate Similarity score each sentence with whole documents		
	scores = {}
	for i in range(len(texts)):
		scores[texts[i]] = pred[i]
		#print score


	#calculate MMR
	#n = 20 * len(sentences) / 100
	limit = 0
	alpha = 0.75 #0.50
	beta = 0.2

	summarySet = []
	posSet = []

	
	summarySet = []
	key_score = list(scores.keys())

	temp_score = key_score[:]

	#max score
	selected = max(scores.items(), key=operator.itemgetter(1))[0]	
	#summarySet.append(selected)
	
	mmr = {}

	while limit < 600:
		for sentence in temp_score:
			mmr[sentence] = alpha * scores[sentence] - (1-alpha) * calculateSimilarity(sentence, summarySet) #- beta*(1/len(sentence.split(" ")))
		
		selected = max(mmr.items(), key=operator.itemgetter(1))[0]
	

		if selected not in summarySet:
			temp_score.remove(selected)
			mmr[selected] = 0
			summarySet.append(selected)
			posSet.append(pos[selected])
			limit = limit + len(selected.split(" "))
			#print(limit)

	
	ret = ""	
	for i  in range(len(summarySet)):
		mind = min(posSet)
		minn = posSet.index(mind)
		posSet[minn] = 1000
		
		ret = ret + " " + summarySet[minn].replace("\n", " ")
	

	return ret


