# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:21:31 2022

@author: Gopinath
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models
import string

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
%matplotlib inline
from matplotlib.pyplot import imread
reviews=pd.read_csv("extract_reviews_test2.csv",encoding='latin1',error_bad_lines=False,index_col=0)
reviews
#error_bad_lines=False means where there is empty lines just ignore that making it False
reviews=[comment.strip() for comment in reviews.comment] # remove both the leading and the trailing characters
reviews=[x for x in reviews if x] # removes empty strings
reviews[0:10]
# Joining the list into one string/text
reviews_text=' '.join(reviews)
reviews_text

#text preprocessing
#Remove Punctuations
no_punc_text=reviews_text.translate(str.maketrans('','',string.punctuation))
no_punc_text

# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk import word_tokenize
text_tokens=word_tokenize(no_punc_text)
print(text_tokens[0:50])
len(text_tokens)

# Remove stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')
sw_list=['I','The','It','A']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# Normalize the data
lower_words=[comment.lower() for comment in no_stop_tokens]
print(lower_words)
# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens)


!python -m spacy download en_core_web_sm
# Lemmatization
import en_core_web_sm
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)

lemmas=[token.lemma_ for token in doc]
print(lemmas)

clean_reviews=' '.join(lemmas)
clean_reviews

#Feature Extaction
#1. Using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
reviewscv=cv.fit_transform(lemmas)
print(cv.vocabulary_)
print(reviewscv.toarray()[150:300])
print(reviewscv.toarray().shape)

#2. CountVectorizer with N-grams (Bigrams & Trigrams)
cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)
print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())

#3. TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matrix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)
print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matrix_ngram.toarray())

#Generate Word Cloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')

# Generate word cloud
STOPWORDS.add('Pron')
wordcloud=WordCloud(width=3000,height=2000,background_color='white',max_words=100,colormap='Set2',stopwords=STOPWORDS).generate(clean_reviews)
plot_cloud(wordcloud)

#Named Entity Recognition (NER)
# Parts of speech (POS) tagging
nlp=spacy.load('en_core_web_sm')
one_block=clean_reviews
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)

for token in doc_block[100:200]:
    print(token,token.pos_)

# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])

# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x: x[1],reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results

# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');

#Emotion Mining - Sentiment Analysis
from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(reviews))
sentences[0:10]

sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df

affin=pd.read_csv('Afinn.csv',sep=',',encoding='Latin-1')
affin

affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores

# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score
# manual testing
calculate_sentiment(text='good service')

# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']

sent_df.sort_values(by='sentiment_value')

# Sentiment score of the whole review
sent_df['sentiment_value'].describe()

# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]

# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]

# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df

# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])

# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

