import csv
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Embedding, LeakyReLU, Flatten, concatenate, GRU, LSTM, Bidirectional, TimeDistributed, dot, multiply, Activation
#from keras.layers.merge import concatenate
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import regularizers

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
tf.keras.backend.clear_session()  # For easy reset of notebook state.

config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.Session(config=config_proto)
set_session(session)

#tf.keras.backend.clear_session()
#tf.config.run_functions_eagerly(True)

from glove_load import load_data
from evaluate import eval_score, eval_str
from diverse import mmr
from test_load import load_test

maxlen = 200
emb_dim = 300

"""
output labels 0(not present in summary) or 1(present in summary)

"""

def calc_test_result(predicted_label, test_label):
    '''
    # Arguments
        predicted test labels, gold test labels 

    # Returns
        accuracy of the predicted labels
    '''
    print("Confusion Matrix :") 
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :") 
    print(classification_report(true_label, predicted_label))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    #return accuracy_score(true_label, predicted_label)


def self_attention(x):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
        
    m = x . transpose(x)
    n = softmax(m)
    o = n . x  
    a = o * x           
       
    return a
        
    '''

    m = dot([x, x], axes=[2,2])
    n = Activation('softmax')(m)
    o = dot([n, x], axes=[2,1])
    a = multiply([o, x])
        
    return a

def cross_modal_attention(x, y):
    
    ''' 
    .  stands for dot product 
    *  stands for elemwise multiplication
    {} stands for concatenation
        
    m1 = x . transpose(y) ||  m2 = y . transpose(x) 
    n1 = softmax(m1)      ||  n2 = softmax(m2)
    o1 = n1 . y           ||  o2 = m2 . x
    a1 = o1 * x           ||  a2 = o2 * y
       
    return {a1, a2}
        
    '''
     
    m1 = dot([x, y], axes=[2, 2])
    n1 = Activation('softmax')(m1)
    o1 = dot([n1, y], axes=[2, 1])
    a1 = multiply([o1, x])

    return a1


print("Loading data.....")
x_train, x_val, y_train, y_val, sp_train, sp_val, vocab_size, word_matrix, sent, summ, los = load_data() 
X_test, P_test, sent_test, lost = load_test()
print("Data Loading Completed")


embedding = Input(shape = (200,))
embeddings = Embedding(input_dim=vocab_size, output_dim=emb_dim, weights=[word_matrix], input_length=maxlen)(embedding)

position = Input(shape = (1,), name = "position")

        
drop_rnn = 0.5
gru_units = 100

rnn_text = Bidirectional(LSTM(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(embeddings)
rnn_text = Flatten()(rnn_text)
rnn_text = Dropout(0.5)(rnn_text) 
merge1 = concatenate([rnn_text, position]) 
dense1 = Dense(100, activation="relu")(merge1) 

conv_text = Conv1D(filters=200, kernel_size=4, activation='relu')(embeddings)
conv_text = MaxPooling1D(pool_size=2)(conv_text)
conv_text = Flatten()(conv_text)
conv_text = Dropout(0.5)(conv_text)
merge2 = concatenate([conv_text, position]) 
dense2 = Dense(100, activation='relu')(merge2) #

merge3 = cross_modal_attention(conv_text, rnn_text)
merge3 = Flatten()(merge3)
merge3 = Dropout(0.5)(merge3)
dense3 = Dense(100, activation='relu')(merge3) #

merge3 = concatenate([dense1, dense2, dense3]) 

fmv = Dense(1,activation="sigmoid")(merge3) 


model = Model(inputs=[embedding, position], outputs=fmv)
model.summary()
#plot_model(model, to_file="MultiView.png")


model.compile(optimizer=RMSprop(lr=0.001), loss="binary_crossentropy", metrics=['accuracy']) #consensus loss + cross entropy loss metrics=["accuracy"] or "binary_accuracy

filepath="weights/BiGRU_CNN.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

"""
print("Fitting model...")
history = model.fit([x_train, sp_train], y_train,
	                    batch_size=200,
	                    epochs=5,
	                    callbacks=callbacks_list,
                    	validation_data=([x_val, sp_val], y_val))
"""

model.load_weights(filepath)

name = "BiGRU_CNN"

scores_dir = "scores/"
output1 = open( scores_dir + "ROUGE_" + str(name) + '.csv', 'w+')
writer1 = csv.writer(output1, delimiter=',')
output_bleu1 = open(scores_dir + "BLEU_" + str(name) + '.csv', 'w+')
writerb1 = csv.writer(output_bleu1, delimiter=',')

output2 = open( scores_dir + "D_ROUGE_" + str(name) + '.csv', 'w+')
writer2 = csv.writer(output2, delimiter=',')
output_bleu2 = open(scores_dir + "D_BLEU_" + str(name) + '.csv', 'w+')
writerb2 = csv.writer(output_bleu2, delimiter=',')


for i in range(len(los)-1):
	print(i)
	prob_list = []
	probability = model.predict([x_val[los[i]:los[i+1]], sp_val[los[i]:los[i+1]]])	

	#print(probability)
	for j in range(len(probability)):
		prob_list.append(probability[j][0])
		
	#print(prob_list, y_val[los[i]:los[i+1]]) 
	#calc_test_result(prob_list, y_val[los[i]:los[i+1]])

	b01,b11,b21,b31, p11,r11,f11, p21,r21,f21, p31,r31,f31 = eval_score(prob_list, sent[los[i]:los[i+1]], summ[i]) 
	writer1.writerow([i, p11,r11,f11, p21,r21,f21, p31,r31,f31])
	writerb1.writerow([i,b01,b11,b21,b31])
	
	divstr = mmr(sent[los[i]:los[i+1]], prob_list)
	b02,b12,b22,b32, p12,r12,f12, p22,r22,f22, p32,r32,f32 = eval_str(divstr, summ[i])
	writer2.writerow([i, p12,r12,f12, p22,r22,f22, p32,r32,f32])
	writerb2.writerow([i,b02,b12,b22,b32])

output1.close()
output_bleu1.close()
output2.close()
output_bleu2.close()

"""
scores_dir = "test_scores/"



for i in range(len(lost)-1):

	item = 1000+i
	print(item)
	output1 = open( scores_dir+ "D0/" + str(item) + '.txt', 'w+')	
	output2 = open(scores_dir + "D1/"+ str(item) + '.txt', 'w+')

	prob_list = []
	probability = model.predict([X_test[lost[i]:lost[i+1]], P_test[lost[i]:lost[i+1]]])	

	for j in range(len(probability)):
		prob_list.append(probability[j][0])
		
	index_sort = np.flip(np.argsort(prob_list)).tolist()
	cs = ""
	count = 0
	candidate = sent_test[lost[i]:lost[i+1]]
	for j in range(len(index_sort)):
		if count <=600:
			index = index_sort.index(len(index_sort)-j-1)
			cs = cs + " " + str(candidate[index]).replace("\n", " ")
			count = len(cs.split(" "))

	sci_text = str(cs)

	div_text = mmr(sent_test[lost[i]:lost[i+1]], prob_list)

	output1.write(sci_text)
	output2.write(div_text)

output1.close()
output2.close()
"""
