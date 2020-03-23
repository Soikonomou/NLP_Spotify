import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re
import gensim
from gensim.models import CoherenceModel
np.random.seed(42)
# initialize the lemmatizer,stemmer.
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
# acquire a list of stopwords from nltk.
stop = stopwords.words('english')
# we will put the data to a pandas dataframe for easy manipulation.
df = pd.read_csv('spotify-public-dataset.csv')
# checking how many raws are empty of each column.
df.isna().sum()
# We notice that 'message_id','author_id','severity' and 'created_at' columns have zero NaN,
# while 'message_type' has 3752 NaN, author_type has 5056 NaN and 'message_body'=14 NaN.
# The message_type and author_type have a lot of NaN values and therefore we will not use them.
# We will use the 'severity' and 'message_body'.
# Since we are going to use the text of the messages we remove the rows that are NaN.
df = df[df['message_body'].notna()]
# Now we will perform pre-processing of the text, including: lowercase the text,
# remove urls, remove non letter characters, tokenize,perform lemmatization and stemming to the text.
df['message_body'] = df.apply(lambda row: row['message_body'].lower(), axis=1)
df['message_body'] = df.apply(lambda row: re.sub(r'http\S+', ' ', row['message_body']), axis=1)
df['message_body'] = df.apply(lambda row: re.sub(r'[^a-zA-Z]', ' ', row['message_body']), axis=1)
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['message_body']), axis=1)
df['tokenized_sents'] = df['tokenized_sents'].apply(
    lambda x: [item for item in x if item not in stop and len(item) > 3 and item != 'spotifycares'])
df['tokenized_sents'] = df['tokenized_sents'].apply(
    lambda x: [stemmer.stem(lemmatizer.lemmatize(item, pos='v')) for item in x])
# put the pre-processed text to a list.
sentences = df['tokenized_sents'].tolist()
# Create a dictionary from 'sentences' containing the number of times a word appears
# in the training set using gensim.corpora.Dictionary and call it 'dictionary'.
dictionary = gensim.corpora.Dictionary(sentences)
# Remove very rare and very common words:
# - words appearing less than 15 times
# - words appearing in more than 10% of all documents
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)
# Create the Bag-of-words model for each text i.e for each text we create a dictionary reporting how many
# words and how many times those words appear. We saved this to 'bow_corpus'.
bow_corpus = [dictionary.doc2bow(doc) for doc in sentences]
# Train our lda model using gensim.models.LdaMulticore and save it to 'lda_model'.
lda_model = gensim.models.LdaMulticore(bow_corpus,
                                       num_topics=10,
                                       id2word=dictionary,
                                       passes=50, workers=2, random_state=42)
# For each topic, we will explore the words occuring in that topic and its relative weight.
different_topics = []
for idx, topic in lda_model.print_topics(-1):
    different_topics.append("Topic: {} \nWords: {}".format(idx, topic))
# We calculate the Coherence score for different number of Topics. The number 10 has the lowest coherence score.
cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
coherence = cm.get_coherence()
# we put the assigned topics with the greatest probability of each text into a list.
topics_with_maxprob = [max(_, key=lambda x:x[1]) for _ in lda_model[bow_corpus]]
only_topics = [i[0] for i in topics_with_maxprob]
# we put the message_ids and severity into lists to use them later.
message_ids = df['message_id'].values.tolist()
severity = df['severity'].values.tolist()
# Below we bring everything together to plot our findings to a file.
# We start by initialize empty lists.
Topic0 = []
Topic1 = []
Topic2 = []
Topic3 = []
Topic4 = []
Topic5 = []
Topic6 = []
Topic7 = []
Topic8 = []
Topic9 = []
severity0 = []
severity1 = []
severity2 = []
severity3 = []
severity4 = []
severity5 = []
severity6 = []
severity7 = []
severity8 = []
severity9 = []
# populate the lists with message_ids and severity.
for i in range(0, len(only_topics)):
    if only_topics[i] == 0:
        Topic0.append(message_ids[i])
        severity0.append(severity[i])
    elif only_topics[i] == 1:
        Topic1.append(message_ids[i])
        severity1.append(severity[i])
    elif only_topics[i] == 2:
        Topic2.append(message_ids[i])
        severity2.append(severity[i])
    elif only_topics[i] == 3:
        Topic3.append(message_ids[i])
        severity3.append(severity[i])
    elif only_topics[i] == 4:
        Topic4.append(message_ids[i])
        severity4.append(severity[i])
    elif only_topics[i] == 5:
        Topic5.append(message_ids[i])
        severity5.append(severity[i])
    elif only_topics[i] == 6:
        Topic6.append(message_ids[i])
        severity6.append(severity[i])
    elif only_topics[i] == 7:
        Topic7.append(message_ids[i])
        severity7.append(severity[i])
    elif only_topics[i] == 8:
        Topic8.append(message_ids[i])
        severity8.append(severity[i])
    elif only_topics[i] == 9:
        Topic9.append(message_ids[i])
        severity9.append(severity[i])
n_low0 = 0
n_medium0 = 0
n_urgent0 = 0
n_low1 = 0
n_medium1 = 0
n_urgent1 = 0
n_low2 = 0
n_medium2 = 0
n_urgent2 = 0
n_low3 = 0
n_medium3 = 0
n_urgent3 = 0
n_low4 = 0
n_medium4 = 0
n_urgent4 = 0
n_low5 = 0
n_medium5 = 0
n_urgent5 = 0
n_low6 = 0
n_medium6 = 0
n_urgent6 = 0
n_low7 = 0
n_medium7 = 0
n_urgent7 = 0
n_low8 = 0
n_medium8 = 0
n_urgent8 = 0
n_low9 = 0
n_medium9 = 0
n_urgent9 = 0
# Count the number of low,medium and urgent for each Topic.
for i in severity0:
    if i == 'low':
        n_low0 += 1
    elif i == 'medium':
        n_medium0 += 1
    elif i == 'urgent':
        n_urgent0 += 1
for i in severity1:
    if i == 'low':
        n_low1 += 1
    elif i == 'medium':
        n_medium1 += 1
    elif i == 'urgent':
        n_urgent1 += 1
for i in severity2:
    if i == 'low':
        n_low2 += 1
    elif i == 'medium':
        n_medium2 += 1
    elif i == 'urgent':
        n_urgent2 += 1
for i in severity3:
    if i == 'low':
        n_low3 += 1
    elif i == 'medium':
        n_medium3 += 1
    elif i == 'urgent':
        n_urgent3 += 1
for i in severity4:
    if i == 'low':
        n_low4 += 1
    elif i == 'medium':
        n_medium4 += 1
    elif i == 'urgent':
        n_urgent4 += 1
for i in severity5:
    if i == 'low':
        n_low5 += 1
    elif i == 'medium':
        n_medium5 += 1
    elif i == 'urgent':
        n_urgent5 += 1
for i in severity6:
    if i == 'low':
        n_low6 += 1
    elif i == 'medium':
        n_medium6 += 1
    elif i == 'urgent':
        n_urgent6 += 1
for i in severity7:
    if i == 'low':
        n_low7 += 1
    elif i == 'medium':
        n_medium7 += 1
    elif i == 'urgent':
        n_urgent7 += 1
for i in severity8:
    if i == 'low':
        n_low8 += 1
    elif i == 'medium':
        n_medium8 += 1
    elif i == 'urgent':
        n_urgent8 += 1
for i in severity9:
    if i == 'low':
        n_low9 += 1
    elif i == 'medium':
        n_medium9 += 1
    elif i == 'urgent':
        n_urgent9 += 1
# Output to a file our results.
text_file = open("Output.txt", "w")
text_file.write(
    'Below you can find the messages clustered in 10 different Topics represented by a list of words.')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[0])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low0))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium0))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent0))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 0:')
text_file.write('\n')
for item in Topic0:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[1])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low1))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium1))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent1))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 1:')
text_file.write('\n')
for item in Topic1:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[2])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low2))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium2))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent2))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 2:')
text_file.write('\n')
for item in Topic2:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[3])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low3))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium3))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent3))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 3:')
text_file.write('\n')
for item in Topic3:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[4])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low4))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium4))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent4))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 4:')
text_file.write('\n')
for item in Topic4:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[5])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low5))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium5))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent5))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 5:')
text_file.write('\n')
for item in Topic5:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[6])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low6))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium6))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent6))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 6:')
text_file.write('\n')
for item in Topic6:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[7])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low7))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium7))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent7))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 7:')
text_file.write('\n')
for item in Topic7:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[8])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low8))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium8))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent8))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 8:')
text_file.write('\n')
for item in Topic8:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.write('\n')
text_file.write(different_topics[9])
text_file.write('\n')
text_file.write('\n')
text_file.write('Number of low severity messages: {}'.format(n_low9))
text_file.write('\n')
text_file.write('Number of medium severity messages: {}'.format(n_medium9))
text_file.write('\n')
text_file.write('Number of urgent severity messages: {}'.format(n_urgent9))
text_file.write('\n')
text_file.write('\n')
text_file.write('Messages ID with the Topic 9:')
text_file.write('\n')
for item in Topic9:
    text_file.write(str(item)+',')
text_file.write('\n')
text_file.close()
