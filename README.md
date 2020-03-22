# NLP_Spotify
In this project I had to cluster together spotify's messages. First, I passed the csv file to a pandas dataframe and examined it. I checked the NaN rows of each attribute and evaluated the importance of each attribute. I concluded that I will use the 'message_body' with Latent Dirichlet Allocation (LDA) to produce a number of Topics with their representative word distribution. The representative words will be very useful when we present our results in the output file since we will be able to infer the Topic's title by these words. The text pre-processing included lowercasing the text, removing urls, removing non-alphanumerical characters, tokenizing the text and performing lemmatization and stemming. After the text pre-processing I used the gensim library to perform LDA and the coherence score to determine the best number of topics. I also used a Bag of Words (BoW) approach instead of TF-IDF because in TF-IDF, heavier weights are given to words that are not as frequent which results in nouns being factored in. That makes it harder to figure out the categories as nouns can be hard to categorize. The 10 topics that the model generated are surprisingly easy to understand (e.g. student premium accounts, family accounts problems, password and login problems or device problems). I used the 'severity' attribute as extra information about each topic to show to the user in the output file. So, the output file showing the Topic's representative words, the number of different severity cases in the Topic and finally the texts clustered together by Topic. I think this kind of representation makes it very easy for the user to easy interpret the results.
