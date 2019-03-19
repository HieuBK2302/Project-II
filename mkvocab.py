from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse, io

f = open("20k_vocab.txt", "r")
cont = []
for l in f:
    l = l.rstrip()
    cont.append(l)
f.close()
#print(cont[1])
print(len(cont))

def mk_vocab(cont):
#	vectorizer = CountVectorizer(min_df = 0.0006, max_df = 0.8)
#	X = vectorizer.fit_transform(cont)
	tfidf_vectorizer = TfidfVectorizer(min_df = 0.0005, max_df = 0.90)
	train = tfidf_vectorizer.fit_transform(cont)
	print(train.shape)
	vocab =  tfidf_vectorizer.get_feature_names()
	io.mmwrite('train.mtx', train)
	print(vocab)
	print(len(vocab))

	vocab = open("vocabulary.txt","w+")
	for w in vocab:
		vocab.write(w+"\n")
	vocab.close()

mk_vocab(cont)





