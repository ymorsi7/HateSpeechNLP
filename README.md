# RumbleNLP

## About

This is my first NLP project. What follows is my initial thoughts on my approach.

- NLP can be used to automatically analyze and structure text (quickly and cost-effectively).
- Using Tf-Idf vectorization, we’ll extract keywords that are classified as hate speech.
- Using logistic regression, we will train a model to classify hate speech using data extracted from the site.

## Scaping

Before scraping, I checked Rumble's site, which stated the following.

> "Systematic retrieval of data or Content from the Rumble Service to create or compile, directly or indirectly, a collection, compilation, library, database or directory without prior written permission from Rumble is prohibited."

Because they specified that "systematic" retrieval of data is prohibited, I needed to scrape the data manually.

As someone who loves automation, this was a tough task, but it needed to be done.

To find sexist content on Rumble, I selected five videos relating to sexism on the platform:

- ["Women Should Not Be in Combat Roles: Change My Mind"](https://rumble.com/v1r74qk-women-should-not-be-in-combat-roles-change-my-mind.html)
- ["The Problem With Modern Women"](https://rumble.com/v1wqypw-the-problem-with-modern-women-w-layah-heilpern-jedediah-bila-live-episode-6.html)
- ["Tucker Carlson Gives CNN Some Tips About Sexism in Hilarious Segment"](https://rumble.com/vfjlp5-tucker-carlson-gives-cnn-some-tips-about-sexism-in-hilarious-segment.html)
- ["Sarah Palin on Kamala Writing Off Criticism of Her As Sexism"](https://rumble.com/vscdbj-sarah-palin-on-kamala-writing-off-criticism-of-her-as-sexism.html)
- ["'B*MBO' MEGHAN MARKLE CLAIMS SHE'S OPPRESSED FOR BEING HOT!"](https://rumble.com/v17l3oh-wheres-kamala-shes-not-doing-anything-shes-lazy-and-she-should-resign.html)