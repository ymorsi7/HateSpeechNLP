import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn. model_selection import train_test_split
from sklearn. tree import DecisionTreeClassifier

import nltk
import re
nltk. download('stopwords')
from nltk. corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk. SnowballStemmer("english")

def get_top_n_words(corpus, n=None):
    '''
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    Args:
        corpus (list): a list of text documents.
        n (int): number of top words to return.
    '''
    assert isinstance(corpus, list), "This must be a list!"
    assert isinstance(n, int), "This must be an integer!"

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(corpus)
    first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[1]
    df_tfidfvectorizer = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

    commentsTF_IDF = df_tfidfvectorizer.sort_values(by=["tfidf"],ascending=False)
    return commentsTF_IDF.head(n)




comments = pd.read_csv('data.csv', encoding='utf-8')
df = pd.DataFrame(comments)
df.drop(['Number'], axis=1, inplace=True) # Drop the Number column (cleaning up the data)
vid1 = vid2 = vid3 = vid4 = vid5 = df



# Let's find the 15 words in each video.


# "Women Should Not Be in Combat Roles: Change My Mind"
vid1 = vid1[vid1.Video == 1]
vid1List = vid1["Comment"].values.tolist()
#print(get_top_n_words(vid1List, 15))



# "The Problem With Modern Women"
vid2 = df[df.Video == 2]
vid2List = vid1["Comment"].values.tolist()
#print(get_top_n_words(vid2List, 15))



# "Tucker Carlson Gives CNN Some Tips About Sexism in Hilarious Segment"
vid3 = df[df.Video == 3]
vid3List = vid1["Comment"].values.tolist()
#print(get_top_n_words(vid3List, 15))



# "WOMAN DEFENDS ANDREW TATE AND ARGUES WITH FEMINISTS AND TRANGENDERS"
vid4 = df[df.Video == 4]
vid4List = vid1["Comment"].values.tolist()
#print(get_top_n_words(vid4List, 15))



# "Massive Feminist March Against Gender Violence in Rome"
vid5 = df[df.Video == 5]
vid5List = vid5["Comment"].values.tolist()
#print(get_top_n_words(vid5List, 15))



# Top 15 Words Overall
df.drop(['Video'], axis=1, inplace=True) # Drop the video column (cleaning up the data)
commentsList = df["Comment"].values.tolist()
#print(get_top_n_words(commentsList, 15))




