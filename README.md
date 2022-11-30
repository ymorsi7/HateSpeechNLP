# NLP Sexism Detection

## About

This is my first NLP project. What follows are my initial thoughts on my approach.

- NLP can be used to automatically analyze and structure text (quickly and cost-effectively).
- Using Tf-Idf vectorization, we’ll extract keywords that are classified as hate speech.
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

## Tf-Idf Vectorization