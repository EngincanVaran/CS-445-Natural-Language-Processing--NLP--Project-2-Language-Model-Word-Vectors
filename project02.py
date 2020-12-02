from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
import  numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import requests
import pickle
from nltk.corpus import stopwords as stw

# def wordClouder():
#     document = ""
#     for i in docs:
#         document += i
#     if stopwords:
#         stopwordsList = stw.words('turkish')
#         wordcloud = WordCloud(
#                 width = dim_size,
#                 height= dim_size,
#                 background_color="white",
#                 stopwords=stopwordsList,
#                 min_font_size=10,
#             ).generate_from_text(document)
#     else:
#         wordcloud = WordCloud(
#                 width = dim_size,
#                 height= dim_size,
#                 background_color="white",
#                 # stopwords=stopwordsList,
#                 min_font_size=10,
#             ).generate_from_text(document)
     
#     wordcloud.to_file("outputs/project2_wordcloud_other.png")

def create_WordCloud(docs, dim_size, output_file_path, mode="TF", stopwords=True):
    if stopwords:
        stopwordsList = stw.words('turkish')
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
    frequency = {}
    myList = []
    for i in docs:
        myList += i.split()
    
    print("List is done: ", len(myList))

    for word in vocabulary:
        frequency[word] = myList.count(word)

    # for word in myList:
    #     if word in vocabulary:
    #         if word in frequency.keys():
    #             frequency[word] += 1
    #         else:
    #             frequency[word] = 1
    
    print("Frequency is done")

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
    return 0

def create_HeapsPlot(docs, output_file_path):
    return 0

def create_LanguageModel(docs, model_type="MLE", ngram=3):
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


wordcloud_outputfile = "outputs/project2_wordcloud.png"
create_WordCloud(doc[:100],800,wordcloud_outputfile,mode="TF",stopwords=True)
print("WordCloud function worked!")