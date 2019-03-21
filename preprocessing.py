import re 
import os
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  
from nltk.tokenize import sent_tokenize, word_tokenize  
from sklearn.feature_extraction.text import CountVectorizer

path = "20_newsgroups"

def ReadFile(file1):
    print(file1)
    with open(file1, "rb") as f:
        cont = f.read()
        cont = cont.decode('utf-8', 'ignore')
    f.close()
    arrays = cont.split()
    # remove hypertext and number, lower, remove stopwords, 
    array1 = []
    stemmer = PorterStemmer()
    for word in arrays:
        word = word.lower() 
        word = re.sub(r'[^a-z]', '', word)
        if(word) not in (stopwords.words('english')):
            if( len(word) < 10 and len(word) > 2):
                array1.append(stemmer.stem(word))
    tf = np.unique(array1, return_counts = True)[1].tolist()
    value = np.unique(array1, return_counts = True)[0].tolist()
    str = ' '.join(value)
    return str
    return cont

cont = ""
label = ""
FJoin = os.path.join
dirs = [FJoin(path, f) for f in os.listdir(path)]

for i in range(0,len(dirs)): 
    d = dirs[i]  
    files = [FJoin(d, f) for f in os.listdir(d)]
    for j in range(0,len(files)): 
        string = ReadFile(files[j])
        label = label+str(i)+"\n"
        string = string + "\n"
        cont = cont + string
l = open("label.txt", "w+")
l.write(label)
l.close() 
f = open("20k_vocab.txt", "w+")
f.write(cont)
f.close()




