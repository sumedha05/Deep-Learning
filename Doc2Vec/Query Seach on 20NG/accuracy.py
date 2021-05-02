####CODE TO CALCULATE ACCURACY OF A MODEL

#########################################################
# FORMAT OF FILES
# TEST FILE - CSV - LABEL , DOCUMENT
# TOP 10 RESULT FILE - LABEL, DOCUMENT
########################################################
from gensim.models import Word2Vec
from sklearn import metrics
from nltk.corpus import stopwords
import numpy as np
import sys
import os
from pprint import pprint
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pickle
from random import shuffle
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
import sys
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"
from collections import namedtuple
import nltk
import re 
from collections import defaultdict
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec


#fetch data from 20NG
parent_path = os.getcwd()
child_path = "20news-bydate-train"
path = os.path.join(parent_path, child_path) 
os.chdir(path)
labels=[]
docs = []
for subdirectory in os.listdir(os.getcwd()):
    sub_path = os.path.join(path,subdirectory)
    #print("Sub directory : ",subdirectory)
    #print(sub_path)
    os.chdir(sub_path)
    for textfile in os.listdir(os.getcwd()):
        textfilepath = os.path.join(sub_path,textfile)
        with open(textfilepath, 'rb') as f:
            content = f.read()
            docs.append(content)
            labels.append(subdirectory)
    #print(docs[-1])
    os.chdir(path)

os.chdir(parent_path)


print(len(docs))
#docs = docs[:200]


#preprocessing on data
def preprocessing(text,stop=True, sent=False, stem=False):
    # Remove punctuations
    #print(text)
    exclude = set(string.punctuation)
    #exclude.append = set((string.digits)
    text = str(text)
    #Remove newline charecter from text
    text = text.replace(r'\n', '\n')
    text = text.replace(r'\t', '\t')
    #text = text.replace('Lines:', '\t')
    #Remove header : from till subject  Re
    subject_idx = text.find("Subject: Re")
    if(subject_idx!=-1):
        text = text[subject_idx+12:]
    

    #Remove header : from till subject (no re)
    subject_idx = text.find("Subject:")
    if(subject_idx!=-1):
        text = text[subject_idx+9:] 

    subject_idx = text.find("Lines:")
    if(subject_idx!=-1):
        text = text[subject_idx+10:]   

    subject_idx = text.find("writes:")
    if(subject_idx!=-1):
        text = text[subject_idx+8:]  
    #print(text)
    #remove punctuations
    text = ''.join(ch for ch in text if ch not in exclude) 
    #print(stopwords.words('english'))
    #print(text)
    tokens = word_tokenize(text)
    if stop:
        tokens = [word.lower() for word in tokens]
        customstop = ['nntppostinghost','aaa','abc','wwii']
        stop = stopwords.words('english')
        tokens =[word for word in tokens if word not in stop ]
        tokens =[word for word in tokens if word not in customstop]
        
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    if sent:
        tokens = ' '.join(tokens)
    remove_list = ['.','0','1','2','3','4','5','6','7','8','9','@','#','*','!','$']
    toklist=[]
    for tok in tokens:
        f=1

        for ch in tok:
            if ch in remove_list:
                #print("here ",ch)
                f=0
                break
        #print(tok)
        if f==1 and len(tok)>2 and len(tok)<17:
            #print(tok)
            toklist.append(tok)
    tokens=toklist
    return tokens

#Preprocessing for Doc2Vec 
def clean_docs(articles):
    clean = []
    for article in articles:
        clean.append(preprocessing(article,stop=True,sent=False, stem=False))
        #sys.exit(1)
    return clean
    
#to convert to a form that doc2vec model can accept using tagged documents
def convert_newsgroup(docs):
    #global doc_count500
    tagged_documents = []
    
    for i,v in enumerate(docs):
        label = '%s'%(i)
        tagged_documents.append(TaggedDocument(v, [label]))
    return tagged_documents

###### REMOVE 50-500 FREQ WORDS
def getfrequency(docs):
    for doc in docs:
        for token in doc:
            frequency[token] += 1
            #print("token ",token," freq ",frequency[token])
    for word in frequency.keys():
        if frequency[word]>70 and frequency[word]<5000:
            vocab.append(word)
    processed_corpus = [[token for token in doc if frequency[token] >70 and frequency[token]<5000] for doc in docs]
    
    #print(frequency)
    return processed_corpus

#print(docs[7])
vocab = []
docs_list= clean_docs(docs)
frequency = defaultdict(int)
processed_corpus = getfrequency(docs_list)
#print(processed_corpus)
#print("here")
#print(processed_corpus)
tagged_data = convert_newsgroup(processed_corpus)
print(len(vocab))
vocab = sorted(vocab)
print(vocab)
#if you donot want this preprocessing, use the below mentioned line and remove the above one
#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(docs)]
    
dm_model =  Doc2Vec(dm=1, dm_mean=1, sample=1e-5, vector_size=200,epochs = 100, negative=5, hs=0,window=5, min_count=5, workers=cores)

#dm_model.load_word2vec_format('/home/skillachie/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
dm_model.build_vocab(tagged_data)  # PV-DM/concat requires one special NULL word so it serves as template


print("TRAINING START")
##### TRAINING
dm_model.train(tagged_data,total_examples=dm_model.corpus_count,epochs=dm_model.epochs)       
dm_model.save("dm20ng.model")
print("TRAINING END")
print('Model Saved')
 

#### TEST ACCURACY
def getaccuracy():
    avg=0.0
    for text in processed_corpus:
        v1 = dm_model.infer_vector(text)
        similar_doc = dm_model.dv.most_similar([v1],topn=1)
        avg+=similar_doc[0][1]
    print("Accuracy of model is : ", (avg/len(processed_corpus)))
getaccuracy()
print("Loss is ",  dm_model.get_latest_training_loss())


####GET TEST DOCUMENTS : 
#fetch data from 20NG
parent_path = os.getcwd()
child_path = "20news-bydate-test"
path = os.path.join(parent_path, child_path) 
os.chdir(path)
labelstest=[]
docstest = []
for subdirectory in os.listdir(os.getcwd()):
    sub_path = os.path.join(path,subdirectory)
    #print("Sub directory : ",subdirectory)
    #print(sub_path)
    os.chdir(sub_path)
    for textfile in os.listdir(os.getcwd()):
        textfilepath = os.path.join(sub_path,textfile)
        with open(textfilepath, 'rb') as f:
            content = f.read()
            docstest.append(content)
            labelstest.append(subdirectory)
    #print(docs[-1])
    os.chdir(path)

os.chdir(parent_path)
print("len is test data",len(docstest))
######PRE PROCESS TEST DATA
docs_list= clean_docs(docstest)
#frequency = defaultdict(int)
processed_corpus_test = getfrequency(docs_list)

query_list = []

for idx in range(0,len(labelstest)):
    tokens = processed_corpus_test[idx]
    tokens = ' '.join(tokens)
    #print(tokens)
    query_list.append(str(tokens))

query_idx = 0
total_precision = 0
for query in query_list:
    ###### get accuracy one by one
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data)
    similar_doc = dm_model.dv.most_similar([v1],topn=1)
    ###### get accuracy from top 10 in all
    precision = 0
    for index in range(1):
        idx = int(similar_doc[index][0])
        if labelstest[query_idx]==labels[idx]:
            precision = precision + 1
    #print("precision for 1 is  ",precision)
    precision = precision
    total_precision = total_precision + precision
    query_idx = query_idx + 1
total_precision = total_precision/len(query_list)
print("PRECISION 1 OF MODEL IS ",total_precision)

query_idx = 0
total_precision = 0
for query in query_list:
    ###### get accuracy one by one
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data)
    similar_doc = dm_model.dv.most_similar([v1],topn=3)
    ###### get accuracy from top 10 in all
    precision = 0
    for index in range(3):
        idx = int(similar_doc[index][0])
        if labelstest[query_idx]==labels[idx]:
            precision = precision + 1
    #print("precision is ",precision)
    precision = precision/3
    total_precision = total_precision + precision
    query_idx = query_idx + 1
total_precision = total_precision/len(query_list)
print("PRECISION 3 OF MODEL IS ",total_precision)

query_idx = 0
total_precision = 0
for query in query_list:
    ###### get accuracy one by one
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data)
    similar_doc = dm_model.dv.most_similar([v1],topn=5)
    ###### get accuracy from top 10 in all
    precision = 0
    for index in range(5):
        idx = int(similar_doc[index][0])
        if labelstest[query_idx]==labels[idx]:
            precision = precision + 1
    #print("precision is ",precision)
    precision = precision/5
    total_precision = total_precision + precision
    query_idx = query_idx + 1
total_precision = total_precision/len(query_list)
print("PRECISION 5 OF MODEL IS ",total_precision)

query_idx = 0
total_precision = 0
for query in query_list:
    ###### get accuracy one by one
    test_data = word_tokenize(query.lower())
    v1 = dm_model.infer_vector(test_data)
    similar_doc = dm_model.dv.most_similar([v1],topn=10)
    ###### get accuracy from top 10 in all
    precision = 0
    for index in range(10):
        idx = int(similar_doc[index][0])
        if labelstest[query_idx]==labels[idx]:
            precision = precision + 1
    #print("precision is ",precision)
    precision = precision/10
    total_precision = total_precision + precision
    query_idx = query_idx + 1
total_precision = total_precision/len(query_list)
print("PRECISION 10 OF MODEL IS ",total_precision)
