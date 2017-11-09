import nltk
from nltk.corpus import sentiwordnet as swn
import csv
import pickle
from sklearn.externals import joblib
import numpy as np
import scipy as sp
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier


###########REPLY TEMPLATES########
def pick_template(fname):
	templates = []
	with open(fname) as f:
		for line in f:
			line = line.strip()
			templates.append(line)
	length = len(templates)-1
	index = randint(0, length)
	rep_temp = templates[index]

	return rep_temp
###

#REPLY TYPE 1: Topical
def rep_topical(topic):
	rep = None
	rep_temp = pick_template('./data/templates_topic')
	rep = rep_temp.replace("_",topic)
	return rep
###

###REPLY TYPE 2: Very long post
def rep_long():
	rep = pick_template('./data/templates_long')
	return rep
###

###REPLY TYPE 3: 
def find_author_attack(tags):
	attack = None
	s_found = False
	for t in tags:
		word = t[0].lower()
		if word =='\'re' or word == 'are':
			s_found = True
			pass
		if s_found == True:
			if t[1] == '.':
				break
			if t[1] == 'NN' or t[1] == 'NNS':
				attack = t[0]
				break

	return attack

def find_merit_attack(tags):
	attack = None
	s_found = False
	for t in tags:
		word = t[0].lower()
		#if t[1] == 'VBZ':
		if word == '\'s' or word == 'is':
			s_found = True
			pass
		if s_found == True:
			if t[1] == '.':
				break
			if t[1] == 'VBN' or t[1] == 'VBG' or t[1] == 'JJ':
				attack = t[0]
				break

	return attack

def check_negative(word):
	negative = False
	synsets = list(swn.senti_synsets(word))
	if len(synsets) == 0:
		negative = True
	else:
		pos = synsets[0].pos_score()
		neg = synsets[0].neg_score()
		if neg > pos:
			negative=True

	return negative

def rep_pattern(tags):
	rep = None
	author = find_author_attack(tags)
	merit = find_merit_attack(tags)
	if author is not None:
		if check_negative(author):
			rep_temp = pick_template('./data/templates_author')
			rep = rep_temp.replace("_",author)

	elif merit is not None:
		if check_negative(merit):
			rep_temp = pick_template('./data/templates_merit')
			rep = rep_temp.replace("_",merit)

	return rep
###

###REPLY TYPE 4: Generic
def rep_generic():
	rep = pick_template('./data/templates_generic')
	return rep

#############TOPIC CLASSIFICATION####
def get_vocab_from_file(fname):
	vocab_set=set()
	i=0
	with open(fname) as f:
		for line in f:
			line = line.strip()
			vocab_set.add(line)

	vocabulary = list(vocab_set)
	return vocabulary

def get_labelled_data(fname):
	X = []
	y = []
	with open(fname) as csvf:
		csvreader = csv.reader(csvf, delimiter=';')
		for row in csvreader:
			if (row[0] != 'abortion' and row[0] != 'evolution'):
				y.append('unknown')
			else:
				y.append(row[0])
			#print(row[1])
			X.append(row[1])
	return X, y

def classify(new_input):
	label = 'NONE'

	#raw, y = get_labelled_data('./data/4forums-topic.csv')
	#neigh = KNeighborsClassifier(n_neighbors=5)
	vocabulary = get_vocab_from_file('./data/dict_abortevol')
	vectorizer = CountVectorizer(vocabulary=vocabulary, strip_accents='ascii')
	#X = vectorizer.transform(raw)
	#neigh.fit(X, y)
	neigh = joblib.load('class.pkl')
	#raw_test = ["Here is an argument about abortion, totes", "Oh really? You suck."]
	raw_test =[new_input]
	#X_test = np.asarray(vectorizer.transform(raw_test))
	#p = neigh.predict_proba(X_test)
	Y_pred = []
	for elem in raw_test:
		#X_test = vectorizer.transform([elem])
		p = neigh.predict_proba(raw_test)
		if p[0][0] >= 0.8:
			Y_pred.append('abortion')
		elif p[0][1] >= 0.8:
			Y_pred.append('evolution')
		else:
			Y_pred.append('unknown')
	label = Y_pred[0]
	return label

####PIPELINE
def reply(quote):
	rep = None
	#STEP 1:
	topic = classify(new_input=quote)
	if topic != 'unknown':
		rep = rep_topical(topic=topic)
	else:
		#STEP 2:
		if len(quote) > 500:
			rep = rep_long()
		else:
			text = nltk.word_tokenize(quote)
			tags = nltk.pos_tag(text)
			rep = rep_pattern(tags)
			#STEP 3:
			if rep is None:
				rep = rep_generic()

	return rep


def main():
	print('Type your input:')
	quote = input()
	print(reply(quote))

if __name__ == "__main__":
    main()