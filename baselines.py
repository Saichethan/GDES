import os
import csv
import sumy
from nltk.translate.bleu_score import sentence_bleu
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #1
from sumy.summarizers.text_rank import TextRankSummarizer #2
from sumy.summarizers.luhn import LuhnSummarizer #3
from sumy.summarizers.sum_basic import SumBasicSummarizer #4
from sumy.summarizers.lsa import LsaSummarizer #5
from sumy.summarizers.kl import KLSummarizer #6
from sumy.summarizers.random import RandomSummarizer #7
from pythonrouge.pythonrouge import Pythonrouge

output = open('BaselinesROUGE.csv', 'w+')
writer = csv.writer(output, delimiter=',')

output_bleu = open('BaselinesBLEU.csv', 'w+')
writerb = csv.writer(output_bleu, delimiter=',')

rp_dir = "/home/reddy/Long_v1/reference/"

plen = []
clen = 0

plen.append(0)

reference_dir = "/home/reddy/Long_v1/reference/"

RP = os.listdir(reference_dir) #len(RP)

sci_text = []
gold_text = []

for i in range(1200, len(RP)):
	file_name = RP[i]
	file_list = open(reference_dir + file_name,'rt') 
	data = csv.reader(file_list)
	data_list = list(data)
	st = ""
	gt = ""

	for j in range(len(data_list)):
		st = st + " " + str(data_list[j][0].replace("\n", " "))
		if int(data_list[j][2]) == 1:
			gt = gt + " " + str(data_list[j][0].replace("\n", " "))
	
	sci_text.append(st)
	gold_text.append(gt)

	

for i in range(len(sci_text)):

	ref_summary = gold_text[i]
	ref_bleu = []
	ref_bleu.append(gold_text[i].split(" "))
	generated1 = ""
	generated2 = ""
	generated3 = ""
	generated4 = ""
	generated5 = ""
	generated6 = ""
	generated7 = ""

	reference = []

	reference.append([[ref_summary]])

	parser = PlaintextParser.from_string(sci_text[i], Tokenizer("english"))

	summarizer1 = LexRankSummarizer()
	summary1 = summarizer1(parser.document,10)


	for sentence in summary1:
		generated1 = generated1 + " " + sentence._text
		
	candidate1 = generated1.split(" ")
	b1 = []
	
	b1.append(sentence_bleu(ref_bleu, candidate1, weights=(1, 0, 0, 0))) #1 gram
	b1.append(sentence_bleu(ref_bleu, candidate1, weights=(0, 1, 0, 0))) #2 gram
	b1.append(sentence_bleu(ref_bleu, candidate1, weights=(0, 0, 1, 0))) #3 gram
	b1.append(sentence_bleu(ref_bleu, candidate1, weights=(0, 0, 0, 1))) #4 gram



	answer1 =[]
	answer1.append([generated1])

	r1 = Pythonrouge(summary_file_exist=False,
                summary=answer1, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score1 = r1.calc_score()

	summarizer2 = TextRankSummarizer()
	summary2 = summarizer2(parser.document,10)


	for sentence in summary2:
		generated2 = generated2 + " " + sentence._text
	
	candidate2 = generated2.split(" ")
	b2 = []
	
	b2.append(sentence_bleu(ref_bleu, candidate2, weights=(1, 0, 0, 0))) #1 gram
	b2.append(sentence_bleu(ref_bleu, candidate2, weights=(0, 1, 0, 0))) #2 gram
	b2.append(sentence_bleu(ref_bleu, candidate2, weights=(0, 0, 1, 0))) #3 gram
	b2.append(sentence_bleu(ref_bleu, candidate2, weights=(0, 0, 0, 1))) #4 gram


	answer2 =[]
	answer2.append([generated2])

	r2 = Pythonrouge(summary_file_exist=False,
                summary=answer2, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score2 = r2.calc_score()

	summarizer3 = LuhnSummarizer()
	summary3 = summarizer3(parser.document,10)


	for sentence in summary3:
		generated3 = generated3 + " " + sentence._text
	
	candidate3 = generated3.split(" ")
	b3 = []
	
	b3.append(sentence_bleu(ref_bleu, candidate3, weights=(1, 0, 0, 0))) #1 gram
	b3.append(sentence_bleu(ref_bleu, candidate3, weights=(0, 1, 0, 0))) #2 gram
	b3.append(sentence_bleu(ref_bleu, candidate3, weights=(0, 0, 1, 0))) #3 gram
	b3.append(sentence_bleu(ref_bleu, candidate3, weights=(0, 0, 0, 1))) #4 gram


	answer3 =[]
	answer3.append([generated3])

	r3 = Pythonrouge(summary_file_exist=False,
                summary=answer3, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score3 = r3.calc_score()

	summarizer4 = SumBasicSummarizer()
	summary4 = summarizer4(parser.document,10)


	for sentence in summary4:
		generated4 = generated4 + " " + sentence._text

	candidate4 = generated4.split(" ")
	b4 = []
	
	b4.append(sentence_bleu(ref_bleu, candidate4, weights=(1, 0, 0, 0))) #1 gram
	b4.append(sentence_bleu(ref_bleu, candidate4, weights=(0, 1, 0, 0))) #2 gram
	b4.append(sentence_bleu(ref_bleu, candidate4, weights=(0, 0, 1, 0))) #3 gram
	b4.append(sentence_bleu(ref_bleu, candidate4, weights=(0, 0, 0, 1))) #4 gram
	

	answer4 =[]
	answer4.append([generated4])

	r4 = Pythonrouge(summary_file_exist=False,
                summary=answer4, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score4 = r4.calc_score()	


	summarizer5 = LsaSummarizer()
	summary5 = summarizer5(parser.document,10)


	for sentence in summary5:
		generated5 = generated5 + " " + sentence._text

	candidate5 = generated5.split(" ")
	b5 = []
	
	b5.append(sentence_bleu(ref_bleu, candidate5, weights=(1, 0, 0, 0))) #1 gram
	b5.append(sentence_bleu(ref_bleu, candidate5, weights=(0, 1, 0, 0))) #2 gram
	b5.append(sentence_bleu(ref_bleu, candidate5, weights=(0, 0, 1, 0))) #3 gram
	b5.append(sentence_bleu(ref_bleu, candidate5, weights=(0, 0, 0, 1))) #4 gram

	
	answer5 =[]
	answer5.append([generated5])

	r5 = Pythonrouge(summary_file_exist=False,
                summary=answer5, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score5 = r5.calc_score()

	summarizer6 = KLSummarizer()
	summary6 = summarizer6(parser.document,10)


	for sentence in summary6:
		generated6 = generated6 + " " + sentence._text

	candidate6 = generated6.split(" ")
	b6 = []
	
	b6.append(sentence_bleu(ref_bleu, candidate6, weights=(1, 0, 0, 0))) #1 gram
	b6.append(sentence_bleu(ref_bleu, candidate6, weights=(0, 1, 0, 0))) #2 gram
	b6.append(sentence_bleu(ref_bleu, candidate6, weights=(0, 0, 1, 0))) #3 gram
	b6.append(sentence_bleu(ref_bleu, candidate6, weights=(0, 0, 0, 1))) #4 gram
	
	answer6 =[]
	answer6.append([generated6])

	r6 = Pythonrouge(summary_file_exist=False,
                summary=answer6, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score6 = r6.calc_score()

	summarizer7 = RandomSummarizer()
	summary7 = summarizer7(parser.document,10)


	for sentence in summary7:
		generated7 = generated7 + " " + sentence._text
	
	candidate7 = generated7.split(" ")
	b7 = []
	
	b7.append(sentence_bleu(ref_bleu, candidate7, weights=(1, 0, 0, 0))) #1 gram
	b7.append(sentence_bleu(ref_bleu, candidate7, weights=(0, 1, 0, 0))) #2 gram
	b7.append(sentence_bleu(ref_bleu, candidate7, weights=(0, 0, 1, 0))) #3 gram
	b7.append(sentence_bleu(ref_bleu, candidate7, weights=(0, 0, 0, 1))) #4 gram

	answer7 =[]
	answer7.append([generated7])

	r7 = Pythonrouge(summary_file_exist=False,
                summary=answer7, reference=reference,
                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                recall_only=False, stemming=True, stopwords=False,
                word_level=True, length_limit=True, length=600,
                use_cf=False, cf=95, scoring_formula='best',
                resampling=True, samples=1, favor=True, p=0.5)
    
	score7 = r7.calc_score()

	
	writerb.writerow([i, b1[0],b1[1],b1[2],b1[3],b2[0],b2[1],b2[2],b2[3], b3[0],b3[1],b3[2],b3[3], b4[0],b4[1],b4[2],b4[3], b5[0],b5[1],b5[2],b5[3], b6[0],b6[1],b6[2],b6[3], b7[0],b7[1],b7[2],b7[3]]) 	

	writer.writerow([i, score1["ROUGE-1-P"], score1["ROUGE-1-R"], score1["ROUGE-1-F"], score1["ROUGE-2-P"], score1["ROUGE-2-R"], score1["ROUGE-2-F"], score1["ROUGE-L-P"], score1["ROUGE-L-R"], score1["ROUGE-L-F"], score2["ROUGE-1-P"], score2["ROUGE-1-R"], score2["ROUGE-1-F"], score2["ROUGE-2-P"], score2["ROUGE-2-R"], score2["ROUGE-2-F"], score2["ROUGE-L-P"], score2["ROUGE-L-R"], score2["ROUGE-L-F"], score3["ROUGE-1-P"], score3["ROUGE-1-R"], score3["ROUGE-1-F"], score3["ROUGE-2-P"], score3["ROUGE-2-R"], score3["ROUGE-2-F"], score3["ROUGE-L-P"], score3["ROUGE-L-R"], score3["ROUGE-L-F"], score4["ROUGE-1-P"], score4["ROUGE-1-R"], score4["ROUGE-1-F"], score4["ROUGE-2-P"], score4["ROUGE-2-R"], score4["ROUGE-2-F"], score4["ROUGE-L-P"], score4["ROUGE-L-R"], score4["ROUGE-L-F"], score5["ROUGE-1-P"], score5["ROUGE-1-R"], score5["ROUGE-1-F"], score5["ROUGE-2-P"], score5["ROUGE-2-R"], score5["ROUGE-2-F"], score5["ROUGE-L-P"], score5["ROUGE-L-R"], score5["ROUGE-L-F"], score6["ROUGE-1-P"], score6["ROUGE-1-R"], score6["ROUGE-1-F"], score6["ROUGE-2-P"], score6["ROUGE-2-R"], score6["ROUGE-2-F"], score6["ROUGE-L-P"], score6["ROUGE-L-R"], score6["ROUGE-L-F"], score7["ROUGE-1-P"], score7["ROUGE-1-R"], score7["ROUGE-1-F"], score7["ROUGE-2-P"], score7["ROUGE-2-R"], score7["ROUGE-2-F"], score7["ROUGE-L-P"], score7["ROUGE-L-R"], score7["ROUGE-L-F"]])

output.close()
output_bleu.close()
