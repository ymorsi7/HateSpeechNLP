# NLP Sexism Detection

## About

This is my first NLP project. What follows are my initial thoughts on my approach.

- NLP can be used to automatically analyze and structure text (quickly and cost-effectively).
- Using Tf-Idf vectorization, we’ll locate the most significant words in each comment section, and overall.
- Using logistic regression, we will train a model to classify hate speech using data extracted from the site.

## Scaping

Before scraping, I checked Rumble's site, which stated the following.

> "Systematic retrieval of data or Content from the Rumble Service to create or compile, directly or indirectly, a collection, compilation, library, database or directory without prior written permission from Rumble is prohibited."

Because they specified that "systematic" retrieval of data is prohibited, I needed to scrape the data manually.

As someone who loves automation, this was a tough task, but it needed to be done.

To find sexist content on Rumble, I selected five videos relating to sexism on the platform:

- [1] ["Women Should Not Be in Combat Roles: Change My Mind"](https://rumble.com/v1r74qk-women-should-not-be-in-combat-roles-change-my-mind.html)
- [2] ["The Problem With Modern Women"](https://rumble.com/v1wqypw-the-problem-with-modern-women-w-layah-heilpern-jedediah-bila-live-episode-6.html)
- [3] ["Tucker Carlson Gives CNN Some Tips About Sexism in Hilarious Segment"](https://rumble.com/vfjlp5-tucker-carlson-gives-cnn-some-tips-about-sexism-in-hilarious-segment.html)
- [4] ["WOMAN DEFENDS ANDREW TATE AND ARGUES WITH FEMINISTS AND TRANGENDERS"](https://rumble.com/v1q566l-woman-defends-andrew-tate-and-argues-with-feminists-and-trangenders-must-wa.html)
- [5] ["Massive Feminist March Against Gender Violence in Rome"](https://rumble.com/v1xflms-massive-feminist-march-against-gender-violence-in-rome.html)

I decided to select the top 50 comments on each video, which totaled 250 comments.

## Data Cleaning

After scraping the data, I needed to clean it. I did this by removing all non-alphanumeric characters, converting all text to lowercase, and removing all stopwords.

## Term Frequency Inverse Document Frequency (TF-IDF)

```python
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

```
