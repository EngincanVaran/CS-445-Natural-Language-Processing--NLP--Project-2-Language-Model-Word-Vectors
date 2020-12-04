import pickle
import re

import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize
from nltk.util import ngrams, bigrams, everygrams
from nltk.corpus import stopwords as stw
from nltk.lm import MLE, KneserNeyInterpolated, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud


def create_WordCloud(docs, dim_size, output_file_path, mode="TF", stopwords=True):
    stopwordsList = set(stw.words('turkish'))
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
    
    global tokenizer, preprocessor, vocabulary

    vectorizer.fit_transform(docs)
    vocabulary = vectorizer.get_feature_names()
    
    preprocessor = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()
    
    rawString = ""
    for i in docs:
        rawString += i

    # Türkiye'nin -> Türkiye
    rawString = re.sub(r'\’[\w]+ ', " " ,rawString)

    if stopwords:
        wc = WordCloud(
            random_state=1,
            width = dim_size,
            height= dim_size,
            background_color="white",
            stopwords=stopwordsList,
            min_font_size=10,
        ).generate(rawString)
    else:
        wc = WordCloud(
            random_state=1,
            width = dim_size,
            height= dim_size,
            background_color="white",
            # stopwords=stopwordsList,
            min_font_size=10,
        ).generate(rawString)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(wc)
    plt.axis("off")
    # plt.show()
    wc.to_file(output_file_path)
    plt.clf()

def create_ZiphsPlot(docs, output_file_path):
    rawString = ""
    for i in range(len(docs)):
        rawString += preprocessor(docs[i])
        # rawString += docs[i]
    
    for i in range(len(docs)):
        # rawString += preprocessor(docs[i])
        rawString += docs[i]
    frequency = {}
    tokens = tokenizer(rawString)
    for token in tokens:
        if token in frequency.keys():
            frequency[token] += 1
        else:
            frequency[token] = 1

    sortedFrequencies = {k: v for k, v in reversed(sorted(frequency.items(), key=lambda item: item[1]))}
    
    freqs = [value for value in sortedFrequencies.values()]
    ranks = range(1,len(sortedFrequencies)+1)
    
    plt.loglog(ranks, freqs, label="Zipf's Law Graph", color="red")
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")
    # plt.show()
    plt.savefig(output_file_path)
    plt.clf()

    plt.scatter(ranks, freqs, label="Zipf's Law Graph", color="green")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig("outputs/normalized_project2_zips.png")
    plt.clf()

def create_HeapsPlot(docs, output_file_path):
    rawString = ""
    for i in range(len(docs)):
        rawString += preprocessor(docs[i])
    
    # Türkiye'nin -> Türkiye
    rawString = re.sub(r'\’[\w]+ ', " " ,rawString)
    # get rid of mi mı
    rawString = re.sub(r' m[iı] ', " ", rawString)

    tokens = tokenizer(rawString)

    corpus = set()
    length = []
    index = range(1,len(tokens)+1)
    for token in tokens:
        corpus.add(token)
        length.append(len(corpus))
    
    plt.plot(index, length, label="Heap's Law Graph", color="red")
    plt.xlabel("Term Occurence")
    plt.ylabel("Vocabulary Size")
    plt.savefig(output_file_path)
    plt.plot(index, index, color="black")
    # plt.show()
    plt.clf()

def create_LanguageModel(docs, model_type="MLE", ngram=3):
    global _ngram
    _ngram = ngram
    tokenized_text = []
    for d in docs:
        text = d
        text = sent_tokenize(text)
        for sent in text:
            temp = []
            for i in word_tokenize(sent):
                temp.append(i.lower())
            tokenized_text.append(temp)

    training_ngrams, vocab = padded_everygram_pipeline(ngram, tokenized_text)

    if model_type == "MLE":
        model = MLE(ngram) #, vocabulary=Vocabulary(vocab))
        model.fit(training_ngrams, vocab)
        # print(model.vocab)
        return model
    elif model_type == "KneserNeyInterpolated" :
        model = KneserNeyInterpolated(ngram)
        model.fit(training_ngrams, vocab) # padded_sents)
        # print(model.vocab)
        return model
    else:
        print("Unkown Model Type")
        return 0

def generate_sentence(model, text):
    detokenize = TreebankWordDetokenizer().detokenize

    sentence_list = []
    perp_list = []

    for i in range(5):
        content = [text]
        while True:
            token = model.generate(text_seed=content)
            if token == '<s>':
                continue
            if token == '</s>':
                break
            content.append(token)

        perp_list.append(model.perplexity(ngrams(content,_ngram)))
        sentence_list.append(detokenize(content))
        print("Done for ", i)

    for i,j in zip(sentence_list, perp_list):
        print(i,j)

    print("\n\n\n")
    index = perp_list.index(min(perp_list))
    
    return (sentence_list[index], perp_list[index])