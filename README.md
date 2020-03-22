# NLP_Spotify
In this project I had to cluster together spotify's messages. First, I passed the csv file to a pandas dataframe and examined it. I checked the NaN rows of each attribute and evaluated the importance of each attribute. I concluded that I would use the 'message_body' with Latent Dirichlet Allocation (LDA) to produce a number of Topics with their representative word distribution. The representative words are very useful when presenting the results in the end since the Topic's title can be inferred easily by these words. The text pre-processing included lowercasing the text, removing urls, removing non-letter characters, tokenizing the text and performing lemmatization and stemming. 
After the text pre-processing I used the gensim library to perform LDA and the coherence score to determine the best number of topics. I also used a Bag of Words (BoW) approach instead of TF-IDF because in TF-IDF, heavier weights are given to words that are not as frequent which results in nouns being factored in. That makes it harder to figure out the categories as nouns can be hard to categorize. The 10 topics that the model generated are surprisingly easy to understand (e.g. student premium accounts, family accounts problems, password and login problems or device problems). I used the 'severity' attribute as extra information about each topic to show to the user in the output file.
The output file shows the Topic's number, the Topic's representative words, the number of different severity cases in the Topic and finally the messages clustered together by Topic. 
For more information and comments please review the 'spotify_analysis.py'.
