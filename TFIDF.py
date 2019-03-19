import numpy as np
import re
import math
import json

doc1 = open("20k_vocab.txt","r")
doc1 = doc1.read()
doc1 = doc1.split("\n")

vocab = open("vocabulary.txt","r")
vocab = vocab.read()
vocab = vocab.split("\n")

'''
Tinh tf : tan so xuat hien 1 tu trong van ban
tf = (so lan xuat hien)/(tong so tu trong van ban)
'''
def TF(wordDict,words):
	tf_dict = {}
	bow_count = len(words)
	#print(bow_count)
	# items tach dung cap ra
	for word, count in wordDict.items():
		tf_dict[word] = count/float(bow_count)
	return tf_dict
#tf_bow1 = TF(wordDict, bow1)
#print(tf_bow1)
# idf la tan so nghich cua 1 tu trong tap van ban
def IDF(bag_list):
	idf_dict ={}
	n = len(bag_list)
	idf_dict = dict.fromkeys(bag_list[0].keys(), 0)
	# dem so lan xuat hien tu trong cac van ban
	for bag in bag_list:
		for word, count in bag.items():
			if count > 0:
				idf_dict[word] += 1

	for word, count in idf_dict.items():
		if count > 0:
			idf_dict[word] = math.log(n/(float(count)))
		if count == 0:
			idf_dict[word] = 0.0
	#print(idf_dict)
	return idf_dict

#idf_dict= IDF()
#print(idf_dict)

def TF_IDT(tf_bow, idfs_dict):
	tfidf = {}
	for word, val in tf_bow.items():
		tfidf[word] = val * idfs_dict[word]
	#print(tfidf)
	return tfidf

#idfs = TF_IDT(tf_bow1, idf_dict)

#wordDict = dict.fromkeys(vocab, 0)
#print(bag["wide"])
'''for bow in doc:
	for word in bow:
		if word in vocab:
			bag[word] += 1
	tf_bow = TF(bag, bow)
	tdf_dict = IDF()
'''
bag_list = []
for doc in doc1:
	words = doc.split()
	wordDict = dict.fromkeys(vocab, 0)
	for word in words:
		if word in vocab:
			wordDict[word] += 1
	bag_list.append(wordDict)

#print(bag_list)
f = open("Tildf.txt","w+")

for doc in doc1:
	words = doc.split()
	wordDict = dict.fromkeys(vocab, 0)
	for word in words:
		if word in vocab:
			wordDict[word] += 1
	tf_bow = TF(wordDict, words)
	tdf_dict = IDF(bag_list)
	tfisd = TF_IDT(tf_bow, tdf_dict)
	for w, v in tfisd.items():
		if v > 0.01:
			f.write(w+" "+str(v)+"\n")
f.close()
