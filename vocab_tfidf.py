from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse, io

contents_train = []

f = open("20k_vocab.txt", "r")

labels = ""
for line in f:
    
    line = line.rstrip()
    line = line.split("________")
    contents_train.append(line[1])
    labels = labels + line[0] + "\n"

f.close()

f = open("label.txt", "w+")

f.write(labels)

f.close()

print("len(contrains_train): ", len(contents_train))

tfidf_vectorizer = TfidfVectorizer(min_df = 0.0005, max_df = 0.90)
train = tfidf_vectorizer.fit_transform(contents_train)
print("train.shape: ", train.shape)
io.mmwrite('train.mtx', train)

voc =  tfidf_vectorizer.get_feature_names()

print(len(voc))


f = open("vocab_train.txt", "w+")

vocab = ""

for i in voc:
    i = i + '\n'
    #print(i)
    vocab = vocab + i

f.write(vocab)

f.close()

'''

contents_test = []
f = open("test.txt", "r")


for line in f:
    
    line = line.rstrip()
    line = line.split("<###>")
    contents_test.append(line[1])
    labels.append(line[0])

f.close()

print("len(contents_test", len(contents_test))

test = tfidf_vectorizer.fit_transform(contents_test)

io.mmwrite('text.mtx', test)

'''





    
