#generates file - ap.txt, ap.dat, vocab.txt

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re
import nltk
nltk.download('wordnet')



import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
#%matplotlib inline
import nltk
import pandas as pd
import random
import string


#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return(" ".join([lemmatizer.lemmatize(w,"v") for w in w_tokenizer.tokenize(text)]))

#limiting the number of data to be fetched
mydata_train = fetch_20newsgroups(subset='train', shuffle=True, remove = ('headers', 'footers', 'quotes'))
mydata_test = fetch_20newsgroups(subset='test', shuffle=True, remove = ('headers', 'footers', 'quotes'))

#now we remove numbers, capital letters etc here
mydata_train = pd.DataFrame({'data': mydata_train.data, 'target': mydata_train.target})
mydata_train.head()

alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

mydata_train['data'] = mydata_train.data.map(alphanumeric).map(punc_lower)
mydata_train["data"] = mydata_train.data.apply(lemmatize_text)
#now vectorize it
count_vect = CountVectorizer(stop_words='english',min_df = 0.0005)

#writing the documents to ap.txt
""" f = open("ap.txt", "w")
list(mydata_train)
print('Training data size:', len(mydata_train['data']))
for text in mydata_train['data']:
    #f.write("<TEXT>\n"+text.encode('utf-8')+"\n</TEXT>\n")
f.close() """

# Printing all the categories
#mydata_train.target_names

f = open("ap.dat", "w")

# Finding frequency of each category
targets, frequency = np.unique(mydata_train.target, return_counts=True)
targets, frequency
#targets_str = np.array(mydata_train.target_names)
#f.write(str(targets_str)+" "+str(frequency))
#print(list(zip(targets_str, frequency)))



X_train_cv = count_vect.fit_transform(mydata_train.data)  # fit_transform learns the vocab and one-hot encodes
X_test_cv = count_vect.transform(mydata_test.data) # transform uses the same vocab and one-hot encodes
print (X_train_cv.shape)

#print(X_train_cv)

#get document term matrix and arrange in format of ap.dat for lda-c
for document in X_train_cv:

    #split by newline
    terms = (str(document)).split('\n')
    #print("--------------------------------")

    #store the data in this and write to file along with length at end
    output_string = ""

    #to calculate number of unique words in document
    length = 0 

    for term in terms:
        term_string = str(term)
        #the term number is between () 
        text_index_start = term_string.find(",")
        text_index_end = term_string.find(")")

        #the frequency occurs after ) , we extract it
        #exception case,if it is not :: in the string, extract it
        if(text_index_start!=-1 and text_index_end!=-1):
            length = length + 1
            term_number = term_string[text_index_start+2:text_index_end]
            term_frequency = term_string[text_index_end+2:]
            output_string = output_string + (term_number+":"+term_frequency+" ")

    #write to file after every document is processed
    f.write(str(length)+" "+output_string+"\n")
f.close()

#write to vocab file 
f = open("vocab.txt", "w")

vocablist = count_vect.get_feature_names()  
for vocab in vocablist:
    f.write(str(vocab.encode('utf-8'))+"\n")

#print(type(X_train_cv))
#print(X_train_cv.toarray())


#tdidf vectorizer

