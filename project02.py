import pickle
from timeit import default_timer as Timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords as stw
from nltk.lm import MLE, KneserNeyInterpolated
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

frequency = {}
vocabulary = []

def create_WordCloud(docs, dim_size, output_file_path, mode="TF", stopwords=True):
    stopwordsList = stw.words('turkish')
    if stopwords:
        if mode == "TF":
            vectorizer = CountVectorizer(stop_words=stopwordsList)
        elif mode == "TFIDF":
            vectorizer = TfidfVectorizer(stop_words=stopwordsList)
        else:
            print("Unknown mode.")
            return
    else:
        if mode == "TF":
            vectorizer = CountVectorizer(stop_words=None)
        elif mode == "TFIDF":
            vectorizer = TfidfVectorizer(stop_words=None)
        else:
            print("Unknown mode.")
            return
    
    vectorizer.fit_transform(docs)
    vocabulary = vectorizer.get_feature_names()
    
    global tokenizor, preprocessor
    preprocessor = vectorizer.build_preprocessor()
    tokenizor = vectorizer.build_tokenizer()

    rawString = ""
    for i in range(len(docs)):
        rawString += preprocessor(docs[i])
    
    # Türkiye'nin -> Türkiye
    rawString = re.sub(r'\'[\w]+', "" ,rawString)
    # get rid of mi mı
    rawString = re.sub(r'm[iı]', "", rawString)

    tokens = tokenizor(rawString)
    for token in tokens:
        if stopwords:
            if token not in stopwordsList:
                if token in frequency.keys():
                    frequency[token] += 1
                else:
                    frequency[token] = 1
        else:
            if token in frequency.keys():
                frequency[token] += 1
            else:
                frequency[token] = 1

    # myList = []
    # for i in docs:
    #     myList += i.split()
    # print("List is done: ", len(myList))
    
    # # start = Timer()
    # # for word in vocabulary:
    # #     frequency[word] = myList.count(word)
    # # end = Timer()
    # # print(end-start)

    # start = Timer()
    # for word in myList:
    #     if word in vocabulary:
    #         if word in frequency.keys():
    #             frequency[word] += 1
    #         else:
    #             frequency[word] = 1
    # end = Timer()
    # print(end-start)
    
    # print("Frequency is done")

    # print(frequency)
    wordcloud = WordCloud(
        random_state=1,
        width = dim_size,
        height= dim_size,
        background_color="white",
        # stopwords=stopwordsList,
        min_font_size=10,
    ).generate_from_frequencies(frequency)
    wordcloud.to_file(output_file_path)

def create_ZiphsPlot(docs, output_file_path):
    sortedFrequencies = {k: v for k, v in reversed(sorted(frequency.items(), key=lambda item: item[1]))}
    
    freqs = [value for value in sortedFrequencies.values()]
    ranks = range(1,len(sortedFrequencies)+1)
    
    plt.loglog(ranks, freqs, label="Zipf's Law Graph")
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")
    plt.savefig(output_file_path)
    plt.clf()

def create_HeapsPlot(docs, output_file_path):
    rawString = ""
    for i in range(len(docs)):
        rawString += preprocessor(docs[i])

    tokens = tokenizor(rawString)
    corpus = set()
    length = []
    index = range(1,len(tokens)+1)
    for token in tokens:
        corpus.add(token)
        length.append(len(corpus))
    
    plt.plot(index, length, label="Heap's Law Graph")
    plt.xlabel("Term Occurence")
    plt.ylabel("Vocabulary Size")
    plt.savefig(output_file_path)
    plt.clf()

def create_LanguageModel(docs, model_type="MLE", ngram=3):
    if model_type == "MLE":
        return MLE(ngram,vocabulary=vocabulary)
    elif model_type == "KneserNeyInterpolated" :
        return KneserNeyInterpolated(ngram,)
    else:
        print("Unkown Model Type")
        return 0

def generate_sentence(model, text="milli"):
    return 0,1

def create_WordVectors(docs, dim_size, model_type, window_size):
    return 0

def use_WordRelationship(model,tuple_list,tuple_test):
    return 0


def read_Docs(path):
    with open(path,'rb') as P:
        Docs = pickle.load(P)
    return Docs

path = "T_sample5000.pkl"
doc = read_Docs(path)

# doc = doc[:100]

wordcloud_outputfile = "outputs/project2_wordcloud.png"
create_WordCloud(doc,800,wordcloud_outputfile,mode="TFIDF",stopwords=True)
print("WordCloud function worked!")

zips_outputfile = "outputs/project2_zips.png"
create_ZiphsPlot(doc,zips_outputfile)
print("Ziph's Law function worked!")

heaps_outputfile = "outputs/project2_heaps.png"
create_HeapsPlot(doc,heaps_outputfile)
print("Heaps' Law function worked!")