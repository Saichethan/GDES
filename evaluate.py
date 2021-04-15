import os
import csv
import sumy
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from pythonrouge.pythonrouge import Pythonrouge


def eval_score(predictions, candidate, reference):

	index_sort = np.flip(np.argsort(predictions)).tolist()

	cs = ""
	count = 0
	

	for i in range(len(index_sort)):
		if count <= 600:
			index = index_sort.index(len(index_sort)-i-1)
			cs = cs + " " + str(candidate[index]).replace("\n", " ")
			count = len(cs.split(" "))
			

	sci_text = str(cs)
	gold_text = str(reference).replace("\n", " ")


	ref_summary = gold_text
	ref_bleu = []
	ref_bleu.append(gold_text.split(" "))

	reference = []
	reference.append([[gold_text]])	
	
	cs_bleu = sci_text.split(" ")

	b = []

	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(1, 0, 0, 0))) #1 gram
	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(0, 1, 0, 0))) #2 gram
	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(0, 0, 1, 0))) #3 gram
	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(0, 0, 0, 1))) #4 gram



	answer =[]
	answer.append([sci_text])

	r = Pythonrouge(summary_file_exist=False,
                summary=answer, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score = r.calc_score()


	
	return b[0],b[1],b[2],b[3], score["ROUGE-1-P"], score["ROUGE-1-R"], score["ROUGE-1-F"], score["ROUGE-2-P"], score["ROUGE-2-R"], score["ROUGE-2-F"], score["ROUGE-L-P"], score["ROUGE-L-R"], score["ROUGE-L-F"] 	


def eval_str(cs, ref):

	sci_text = str(cs).replace("\n", " ")
	gold_text = str(ref).replace("\n", " ")

	
	ref_summary = gold_text
	ref_bleu = []
	ref_bleu.append(gold_text.split(" "))

	reference = []
	reference.append([[gold_text]])	
	
	cs_bleu = sci_text.split(" ")

	b = []

	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(1, 0, 0, 0))) #1 gram
	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(0, 1, 0, 0))) #2 gram
	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(0, 0, 1, 0))) #3 gram
	b.append(sentence_bleu(ref_bleu, cs_bleu, weights=(0, 0, 0, 1))) #4 gram



	answer =[]
	answer.append([sci_text])

	r = Pythonrouge(summary_file_exist=False,
                summary=answer, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score = r.calc_score()


	
	return b[0],b[1],b[2],b[3], score["ROUGE-1-P"], score["ROUGE-1-R"], score["ROUGE-1-F"], score["ROUGE-2-P"], score["ROUGE-2-R"], score["ROUGE-2-F"], score["ROUGE-L-P"], score["ROUGE-L-R"], score["ROUGE-L-F"] 	

