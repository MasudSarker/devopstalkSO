# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:44:55 2021

@author: masud
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 07:10:08 2021

@author: masud
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0#:~:text=The%20concept%20of%20topic%20coherence,scoring%20words%20in%20the%20topic."""

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim import models
from gensim.models import LdaModel, CoherenceModel
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy
from spacy.lang.en import English

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

reviews_datasets = pd.read_csv("F:\\Project\\DevOps_Dataset.csv",encoding="utf8")
#reviews_datasets = reviews_datasets.head(40)
reviews_datasets.dropna()
cview = reviews_datasets['ViewCount'].astype(int)
avgfav = reviews_datasets['FavoriteCount'].astype(int)
avgscore = reviews_datasets['Score'].astype(int)
body = reviews_datasets['Body']
titles = reviews_datasets['Title']
caccepans = reviews_datasets['AcceptedAnsCount']
ansdelay = reviews_datasets['answerdelay']
avgdelay = ansdelay.fillna(0)
#avgdelay = ansdelay*24;
#print(avgdelay)
reviews_datasets.head()


# In[15]:


from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk

import re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer

tokenizer = ToktokTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

# Preprocess the text for vectorization
# - Remove HTML
# - Remove stopwords
# - Remove special characters
# - Convert to lowercase
# - Stemming


# In[25]:


import re

# Convert to list
data = reviews_datasets.Title.values.tolist()

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single and double quotes
data = [re.sub("\'", "", sent) for sent in data]
data = [re.sub('\", "', '', sent) for sent in data]
data = [re.sub('\\"', '', sent) for sent in data]
data = [re.sub('\"', '', sent) for sent in data]
data = [re.sub('[\\:"]', '', sent) for sent in data]

# Remove web links
data = [re.sub(r'^https?:\/\/.*[\r\n]*', '', sent) for sent in data]

#pprint(data[:3])


# In[30]:


# Tokenize words and text clean up
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

#print(data_words[:3])


# In[31]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=75) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=75)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
#print(trigram_mod[bigram_mod[data_words[3]]])


# In[32]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[64]:


data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#spacy.load('en_core_web_sm')

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:2])


# In[65]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])


# In[66]:

#gamma_threshold
# Build LDA model
#minimum_phi_value
#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            # id2word=id2word,
                                            # num_topics=28, 
                                            # random_state=100,
                                            # update_every=1,
                                            # chunksize=100,
                                            # passes=10,
                                            # alpha='auto',
                                            #  eta=.01,
                                            #  eval_every=10,                                            
                                            # per_word_topics=True)


# In[67]:


# Print the Keyword in the 25 topics
#print(lda_model.print_topics())
#doc_lda = lda_model[corpus]


# In[78]:


# Compute Perplexity
#print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
##coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
##coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)


# In[89]:
######################################
# LDA Mallet for Topic Modeling
######################################

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
#mallet_path = 'D:/mallet-2.0.8/bin/mallet' # update this path
#lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=15, id2word=id2word, optimize_interval=10, iterations=1000)

import os
from gensim.models.wrappers import LdaMallet

os.environ['MALLET_HOME'] = 'F:\\mallet-2.0.8'

mallet_path = 'F:\\mallet-2.0.8\\bin\\mallet'
#ldamallet_test = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=dictionary_test)
lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=30, id2word=id2word, optimize_interval=10, iterations=1000, alpha=5)


# In[90]:

    ## https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#16buildingldamalletmodel
    
#lda_mallet.show_topics(num_topics=-1, num_words=20, formatted=False).to_csv("D:\\final_topics_06032021.csv")
# Show Topics
#print(lda_mallet.show_topics(num_topics=-1, num_words=20, formatted=False))

# Compute Coherence Score
#coherence_model_ldamallet = CoherenceModel(model=lda_mallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_ldamallet = coherence_model_ldamallet.get_coherence()
#print('\nCoherence Score: ', coherence_ldamallet)


# In[97]:
###################################
# Devops topic development
###################################

def format_topics_sentences(ldamodel=lda_mallet, corpus=corpus, texts=data_lemmatized, title=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    contents2 = pd.Series(title)
    sent_topics_df = pd.concat([sent_topics_df, contents, contents2], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_mallet, corpus=corpus, texts=body, title=titles)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'texts', 'title']
df_dominant_topic.to_csv("F:\\Project\\final_topics_include_postbody.csv")

#df_dominant_topic.groupby('Keywords')['Text'].sum()
# Show
#print(df_dominant_topic.head(10))


# In[ ]:

    
#Python Topics Modeling
python_topics_modeling = df_dominant_topic.groupby(['Keywords','Dominant_Topic'])['Dominant_Topic'].count().astype(int)
#print(python_topics_modeling)
python_topics_modeling.to_csv("F:Project\\final_topics_modeling.csv")



#######################################
# topic Propularity measure
#######################################

def format_topics_sentences2(ldamodel=lda_mallet, corpus=corpus, avgview=data_lemmatized, avgfav=data_lemmatized, avgscore=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(avgview)
    contents2 = pd.Series(avgfav)
    contents3 = pd.Series(avgscore)
    sent_topics_df = pd.concat([sent_topics_df, contents, contents2, contents3], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences2(ldamodel=lda_mallet, corpus=corpus, avgview=cview, avgfav=avgfav, avgscore=avgscore)



# In[ ]:
   
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'avgview', 'avgfav', 'avgscore']

#print(df_dominant_topic.head())
# Topics Wise average (View)
averag_view_wise_topics = df_dominant_topic.groupby('Keywords').agg({'avgview':'mean', 'avgfav':'mean','avgscore':'mean'})
#averag_view_wise_topics = df_dominant_topic.groupby('Keywords')['avgview'].mean().astype(int)
#print(averag_view_wise_topics)
averag_view_wise_topics.to_csv("F:\\Project\\final_topics_popularity.csv")

#average_wise_topcs.sort_index(ascending =False)

# Topics Wise sum (View)
#df_dominant_topic.groupby('Keywords')['Text'].sum()



#######################################
# topic difficulty measure
#######################################

def format_topics_sentences3(ldamodel=lda_mallet, corpus=corpus, accepans=data_lemmatized, avgdelay=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(accepans)
    contents2 = pd.Series(avgdelay)
    
    sent_topics_df = pd.concat([sent_topics_df, contents, contents2], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences3(ldamodel=lda_mallet, corpus=corpus, accepans=caccepans, avgdelay=avgdelay )



# In[ ]:
   
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'accepans', 'avgdelay']

#print(df_dominant_topic.head())
# Topics Wise average (View)
topics_difficulty = df_dominant_topic.groupby('Keywords').agg({'accepans':'sum', 'avgdelay':'mean'})
#averag_view_wise_topics = df_dominant_topic.groupby('Keywords')['avgview'].mean().astype(int)
#print(averag_view_wise_topics)
topics_difficulty.to_csv("F:\\Project\\final_topics_difficulty.csv")
