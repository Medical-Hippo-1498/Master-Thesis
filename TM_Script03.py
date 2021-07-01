# Objective
	## 1. Pre-process the data
	## 1. Fit a LDA model to the cleaned and pre-processed data
	## 2. Group topics together based on their Hellinger distance
	## 3. Label comments with topics


# Import the data and the modules
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import gensim
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import pyLDAvis.gensim
import nltk
import numpy as np
np.random.seed(2018)

from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.matutils import hellinger
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora, models
from pprint import pprint
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.stem.porter import *
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split

nltk.download('wordnet')

all_stopwords_gensim = STOPWORDS.union(set(['awww', 
                                            'afternoon', 
                                            'mmmm', 
                                            'haha', 
                                            'amazing',
                                            'awesome',
                                            'gorgeous',
                                            'sexy',
                                            'fantastic',
                                            'beautiful',
                                            'perfect',
                                            'favorite',
                                            'great',
                                            'good',
                                            'sweet',
                                            'best',
                                            'super',
                                            'nice',
                                            'xoxo',
                                            'crazy',
                                            'soooo',
                                            'incredible',
                                            'absolute',
                                            'delicioous',
                                            'mmmmm',
                                            'gonna',
                                            'bruh',
                                            'yeah',
                                            'perfect',
                                            'follow',
                                            'sooo',
                                            'yummi',
                                            'hello',
                                            'woooow',
                                            'thank',
                                            "thanks",
                                            "get",
                                            "superb",
                                            "aboslute",
                                            "absolutely",
                                            "cool",
                                            "beautiful",
                                            "maybe",
                                            "damn",
                                            "that",
                                            "perfect",
                                            "shes",
                                            "yummy",
                                            "yummi",
                                            "lmao",
                                            "yooou",
                                            "magnifique",
                                            "love",
                                            "lovely",
                                            "mmmmmmm",
                                            "yoou",
                                            "wooow",
                                            "xoxoxo",
                                            "sooooo",
                                            "wowww",
                                            "yaay",
                                            "ohhh",
                                            "hotttt",
                                            "whoa",
                                            "hehe",
                                            "dang",
                                            "xxxx",
                                            "goddamn",
                                            "sublime",
                                            "fabulous",
                                            "fuckkk",
                                            "gooood",
                                            "xoxo",
                                            "soooooo",
                                            "mmmmmm",
                                            "hahaha",
                                            "xxxxx",
                                            "nicee",
                                            "veeery",
                                            "sexxxxyy",
                                            "delici",
                                            "sensual",
                                            "greatest",                                            
                                            "didnt"
                                            "will",
                                            "cute",
                                            "kind",
                                            "couldn",
                                            "magnific",
                                            "gostosa",
                                            "deliciosa",
                                            "delicia",
                                            "wonderful",
                                            "genuin",
                                            "love",
                                            "instead",
                                            "huge",
                                            "impress",
                                            "delicioso",
                                            "wish",
                                            "hotter",
                                            "begin",
                                            "unbeliev",
                                            "congratul",
                                            "esta",
                                            "como",
                                            "boca",
                                            "buen",
                                            "est",
                                            "quiero",
                                            "youu",
                                            "doesnt",
                                            "beauti",
                                            "bravo",
                                            "brilliant",
                                            "prettiest",
                                            "believ",
                                            "wanna",
                                            "eat",
                                            "astonish",
                                            "yesss",
                                            "worst",
                                            "princess",
                                            "bella",
                                            "aussi",
                                            "superb",
                                            "woow",
                                            "hilari",
                                            "divin",
                                            "yessss",
                                            "have",
                                            "goddess",
                                            "hmmm",
                                            "hermosa",
                                            "para",
                                            "pero",
                                            "hermoso",
                                            "insan",
                                            "later",
                                            "sexiest",
                                            "wasn",
                                            "woah",
                                            "oooh",
                                            "sehr",
                                            "geil",
                                            "damm",
                                            "flawless",
                                            "yall"]))


stemmer = SnowballStemmer("english")

data = pd.read_csv("ReadyForAnalysis.csv")
cols = data.columns

# Add all of the other columns
data_full = pd.DataFrame(data.loc[(data.clean==True) & (data.is_eng==True), ["title", 
                                                                             "views", 
                                                                             "up_votes", 
                                                                             "down_votes", 
                                                                             "percent", 
                                                                             "author",
                                                                             "author_subscriber",
                                                                             "categories",
                                                                             "tags",
                                                                             "production",
                                                                             "description",
                                                                             "duration",
                                                                             "upload_date",
                                                                             "pornstars",
                                                                             "download_urls",
                                                                             "thumbnail_url",
                                                                             "number_of_comment",
                                                                             "url",
                                                                             "error",
                                                                             "repeat_n_times",
                                                                             "comms",
                                                                             "clean",
                                                                             "is_eng"
                                                                            ]])

data_text = data_full = pd.DataFrame(data.loc[(data.clean==True) & (data.is_eng==True), ["comms"]])

data_text = data_text.dropna(subset=['comms'])
data_text = data_text.reset_index(drop=True)
data_text['index'] = data_text.index

documents = data_text


# PT1: Data pre-processing
    ## -Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
    ## -Words are stemmed — words are reduced to their root form. (Happiness --> Happy)
    ## -Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
    ## -Words that have fewer than 3 characters are removed.
    ## -All stopwords are removed.
	
## Lemmatize and stemming function
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in all_stopwords_gensim and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


## Run the function over the whole dataset
processed_docs = documents["comms"].map(preprocess)

## Also store the results in the full dataframe
processed_coms = processed_docs
data_full["processed_coms"] = processed_coms

## Remove empty comments
print(len(data_full))

	#Remove empty rows
data_full = data_full.dropna(subset=['processed_coms'])
data_full = data_full.reset_index(drop=True)

print(len(data_full))

	#Remove empty comments
data_full = data_full[data_full['processed_coms'].map(lambda d: len(d)) > 0]
data_full = data_full.reset_index(drop=True)

print(len(data_full))

	#Assign the new values to processed_docs
processed_docs = data_full["processed_coms"]


# Bigrams
## Build the bigram models
bigram = gensim.models.phrases.Phrases(processed_docs, min_count=3, threshold=10)
bigram_mod = gensim.models.phrases.Phraser(bigram)

processed_docs_bigram = {"bigram": []}
for i in range(len(processed_docs)) :
    big = bigram[processed_docs[i]]
    
    processed_docs_bigram["bigram"].append(big)

processed_docs_bigram = pd.DataFrame(processed_docs_bigram)

processed_docs = processed_docs_bigram["bigram"]
data_full["processed_coms"] = processed_docs


# Save the processed data
data_full.to_csv(r'Processed_Docs_full.csv')


# PT2: Training the model
## We create a custom dictionary based on our dataset
dictionary = gensim.corpora.Dictionary(processed_docs)

### We filter out words that:
    #### -Are present in less than 15 comments (too rare)
    #### -In more than 50% of the comments (too numerous --> stopwords)
    #### - After the first two steps, we only keep the first 100k tokens/words (further filtering of stopwords)
	
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# Model evaluation and tuning (sensitivity analysis): coherence = C_v
	#Compute coherence score for range of topics
Sensitivity_analysis = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                        'Perplexity': []
                }

	#Topics(K):
topics_range = range(3,66,1)

	#Alpha:
alpha = list(np.arange(0.01, 1, 0.3))

	#Beta:
beta = list(np.arange(0.01, 1, 0.3))

	#Number of passes:
num_passes = 2

	#Create training and holdout data
train, test = train_test_split(processed_docs, test_size=0.2)

	# Training dictionary and BoW
dictionary = gensim.corpora.Dictionary(train)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in train]



	#Loop over the range of values
for k in topics_range:
    print(k)
    print("topics out of")
    print(max(topics_range))
    print("-------------------------------------")
    for a in alpha:
        for b in beta:
            
            lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=num_passes,
                                           alpha=a,
                                           eta=b)
            coherence_model_lda = CoherenceModel(model=lda_model, 
                                                       texts=test, 
                                                       dictionary=dictionary, 
                                                       coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            
            perplexity_lda = lda_model.log_perplexity(bow_corpus)
            
            Sensitivity_analysis["Topics"].append(k)
            Sensitivity_analysis["Alpha"].append(a)
            Sensitivity_analysis["Beta"].append(b)
            Sensitivity_analysis["Coherence"].append(coherence_lda)
            Sensitivity_analysis["Perplexity"].append(perplexity_lda)

			
	#Save the results internally
Sensitivity_analysis_pd = pd.DataFrame.from_dict(Sensitivity_analysis)

	#Save the results externally
Sensitivity_analysis_pd.to_csv('Sensitivity_analysis.csv', index=False)


## Additional loop for alpha symmetric
	#Compute coherence score for range of topics
Sensitivity_analysis = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                        'Perplexity': []
                }

	#Topics(K):
topics_range = range(3,66,1)

	#Alpha:
alpha = "symmetric"

	#Beta:
beta = list(np.arange(0.01, 1, 0.3))

	#Number of passes:
num_passes = 2

	#Create training and holdout data
train, test = train_test_split(processed_docs, test_size=0.2)

	# Training dictionary and BoW
dictionary = gensim.corpora.Dictionary(train)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in train]


	#Loop over the range of values
for k in topics_range:
    print(k)
    print("topics out of")
    print(max(topics_range))
    print("-------------------------------------")
    
    for b in beta:
            
        lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=num_passes,
                                           alpha=alpha,
                                           eta=b)
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                                       texts=test, 
                                                       dictionary=dictionary, 
                                                       coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
            
        perplexity_lda = lda_model.log_perplexity(bow_corpus)
            
        Sensitivity_analysis["Topics"].append(k)
        Sensitivity_analysis["Alpha"].append(alpha)
        Sensitivity_analysis["Beta"].append(b)
        Sensitivity_analysis["Coherence"].append(coherence_lda)
        Sensitivity_analysis["Perplexity"].append(perplexity_lda)
            
    #Save the results internally
Sensitivity_analysis_pd = pd.DataFrame.from_dict(Sensitivity_analysis)

	#Save the results externally
Sensitivity_analysis_pd.to_csv('Sensitivity_analysis_ALPHAsymmetric_3-65.csv', index=False)


## Additional loop for Alpha asymmetric
	#Compute coherence score for range of topics
Sensitivity_analysis = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                        'Perplexity': []
                }

	#Topics(K):
topics_range = range(3,66,1)

	#Alpha:
alpha = "asymmetric"

	#Beta:
beta = list(np.arange(0.01, 1, 0.3))

	#Number of passes:
num_passes = 2

	#Create training and holdout data
train, test = train_test_split(processed_docs, test_size=0.2)

	# Training dictionary and BoW
dictionary = gensim.corpora.Dictionary(train)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in train]


	#Loop over the range of values
for k in topics_range:
    print(k)
    print("topics out of")
    print(max(topics_range))
    print("-------------------------------------")
    
    for b in beta:
            
        lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=num_passes,
                                           alpha=alpha,
                                           eta=b)
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                                       texts=test, 
                                                       dictionary=dictionary, 
                                                       coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
            
        perplexity_lda = lda_model.log_perplexity(bow_corpus)
            
        Sensitivity_analysis["Topics"].append(k)
        Sensitivity_analysis["Alpha"].append(alpha)
        Sensitivity_analysis["Beta"].append(b)
        Sensitivity_analysis["Coherence"].append(coherence_lda)
        Sensitivity_analysis["Perplexity"].append(perplexity_lda)
    
	#Save the results internally
Sensitivity_analysis_pd = pd.DataFrame.from_dict(Sensitivity_analysis)

	#Save the results externally
Sensitivity_analysis_pd.to_csv('Sensitivity_analysis_ALPHAasymmetric_3-65.csv', index=False)

## Additional loop for Alpha and Beta symmetric
	#Compute coherence score for range of topics
Sensitivity_analysis = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                        'Perplexity': []
                }

	#Topics(K):
topics_range = range(3,66,1)

	Alpha:
alpha = "symmetric"

	#Beta:
beta = "symmetric"

	#Number of passes:
num_passes = 2

	#Create training and holdout data
train, test = train_test_split(processed_docs, test_size=0.2)

	# Training dictionary and BoW
dictionary = gensim.corpora.Dictionary(train)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in train]


	#Loop over the range of values
for k in topics_range:
    print(k)
    print("topics out of")
    print(max(topics_range))
    print("-------------------------------------")
            
    lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=num_passes,
                                           alpha=alpha,
                                           eta=beta)
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                                       texts=test, 
                                                       dictionary=dictionary, 
                                                       coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
            
    perplexity_lda = lda_model.log_perplexity(bow_corpus)
            
    Sensitivity_analysis["Topics"].append(k)
    Sensitivity_analysis["Alpha"].append(alpha)
    Sensitivity_analysis["Beta"].append(beta)
    Sensitivity_analysis["Coherence"].append(coherence_lda)
    Sensitivity_analysis["Perplexity"].append(perplexity_lda)
    
	#Save the results internally
Sensitivity_analysis_pd = pd.DataFrame.from_dict(Sensitivity_analysis)

	#Save the results externally
Sensitivity_analysis_pd.to_csv('Sensitivity_analysis_symmetric_3-65.csv', index=False)

## Additional loop for Beta symmetric

	#Compute coherence score for range of topics
Sensitivity_analysis = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                        'Perplexity': []
                }

	#Topics(K):
topics_range = range(3,66,1)

	#Alpha:
alpha = list(np.arange(0.01, 1, 0.3))

	#Beta:
beta = "symmetric"

	#Number of passes:
num_passes = 2

	#Create training and holdout data
train, test = train_test_split(processed_docs, test_size=0.2)

	# Training dictionary and BoW
dictionary = gensim.corpora.Dictionary(train)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in train]

	#Loop over the range of values
for k in topics_range:
    print(k)
    print("topics out of")
    print(max(topics_range))
    print("-------------------------------------")
    for a in alpha:
            
        lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=num_passes,
                                           alpha=a,
                                           eta=beta)
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                                       texts=test, 
                                                       dictionary=dictionary, 
                                                       coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
            
        perplexity_lda = lda_model.log_perplexity(bow_corpus)
            
        Sensitivity_analysis["Topics"].append(k)
        Sensitivity_analysis["Alpha"].append(a)
        Sensitivity_analysis["Beta"].append(beta)
        Sensitivity_analysis["Coherence"].append(coherence_lda)
        Sensitivity_analysis["Perplexity"].append(perplexity_lda)
    
	#Save the results internally
Sensitivity_analysis_pd = pd.DataFrame.from_dict(Sensitivity_analysis)

	#Save the results externally
Sensitivity_analysis_pd.to_csv('Sensitivity_analysis_BETAsymmetric_3-65.csv', index=False)	


# Generate best model based on sensitivity analysis
# Dictionary and BoW
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

k = 57
num_passes = 10
a = 0.01
b = 0.61

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=num_passes,
                                           alpha=a,
                                           eta=b)


# Save model
	##Save model to disk
# lda_model.save('lda_model.model')

	## To load the model
# lda_model_tfidf =  models.LdaModel.load('lda_model_tfidf.model')


# PT3: Topic labelling
# Record the distance between topics

## Record the words in each topic
num_topics = 57
num_words = 20

Words_in_topics = {'Topics': [],
                 'Words': [],
                 'Probability': []}

words = lda_model.show_topics(formatted=False,num_words=num_words, num_topics = num_topics)

for i in range(num_topics):
    for n in range(num_words):
        word_topic = words[i][1][n][0]
        word_prob = words[i][1][n][1]

        Words_in_topics["Topics"].append(i)
        Words_in_topics["Words"].append(word_topic)
        Words_in_topics["Probability"].append(word_prob)

Words_in_topics = pd.DataFrame(Words_in_topics)

## Save the words in each topic
Words_in_topics.to_csv("words_in_topics.csv")


## Compute the Hellinger distance
	#Processing function for Hellinger distance
topic_distribution = {"Topics": [],
              "Prob_distribution": []}

for i in range(num_topics):
    words_topic = Words_in_topics.loc[(Words_in_topics.Topics==i), ["Words"]].values.tolist()
    words_topic = sum(words_topic, [])
    
    prob_dist = lda_model.id2word.doc2bow(words_topic)  
    
    topic_distribution["Topics"].append(i)
    topic_distribution["Prob_distribution"].append(prob_dist)

topic_distribution = pd.DataFrame(topic_distribution)


	# Hellinger distance
topic_distances = {"from": [],
                  "to": [],
                  "Hellinger_distance": []}

for i in range(num_topics):
    for y in range(num_topics):
        hellinger_distance = hellinger(sum(topic_distribution.loc[(topic_distribution.Topics == i), ["Prob_distribution"]].values.tolist(),[]),
                                      sum(topic_distribution.loc[(topic_distribution.Topics == y), ["Prob_distribution"]].values.tolist(),[]))
        
        topic_distances["from"].append(i)
        topic_distances["to"].append(y)
        topic_distances["Hellinger_distance"].append(hellinger_distance)

topic_distances = pd.DataFrame(topic_distances)

## Save the distance between topics
topic_distances.to_csv('Topic_distances.csv', index=False)


#Group the topics together based on their Hellinger distance and summarise them with a label based on their word distribution
Topic1 = "Duration"
Topic2 = "Looks"
Topic3 = "Cluster3"
Topic4 = "Cluster4"
Topic5 = "Cluster5"
Topic6 = "Female_attribute"
Topic7 = "Video"
Topic8 = "Skills"
Topic9 = "Intercourse"
Topic10 = "Orgasm"


# PT4: Comments labelling
## Load the data
final_data = pd.read_csv("ReadyForAnalysis.csv")
cols = data.columns

	# Add all of the other columns
final_data = pd.DataFrame(final_data.loc[(final_data.clean==True) & (final_data.is_eng==True), ["title", 
                                                                             "views", 
                                                                             "up_votes", 
                                                                             "down_votes", 
                                                                             "percent", 
                                                                             "author",
                                                                             "author_subscriber",
                                                                             "categories",
                                                                             "tags",
                                                                             "production",
                                                                             "description",
                                                                             "duration",
                                                                             "upload_date",
                                                                             "pornstars",
                                                                             "download_urls",
                                                                             "thumbnail_url",
                                                                             "number_of_comment",
                                                                             "url",
                                                                             "error",
                                                                             "repeat_n_times",
                                                                             "comms",
                                                                             "clean",
                                                                             "is_eng"
                                                                            ]])

final_data = final_data.dropna(subset=['comms'])
final_data = final_data.reset_index(drop=True)
final_data['index'] = final_data.index

## Label comments
	# Create the labelling function
def labelling(com):

    #Process the new document
    com = preprocess(com)
    
    #Compute matching topic
    topics = lda_model.show_topics(formatted=True, num_topics=57, num_words=10)
    label = pd.DataFrame([(el[0], round(el[1],2), topics[el[0]][1]) for el in lda_model[dictionary.doc2bow(com)]], 
             columns=['topic', 'weight', 'words in topic'])
    label = label.sort_values(by=['weight'], ascending = False)
    label = label.reset_index(drop=True)
    
    #Record topic and weight
    topic_n_weight = []
    
    topic = label.loc[0, "topic"]
    weight = label.loc[0, "weight"]
    
    topic_n_weight.append(topic)
    topic_n_weight.append(weight)
    
    return(topic_n_weight)
	
## Apply labelling function to every document in the dataset
	#Create columns
final_data["topic_nb"] = range(len(final_data))
# final_data["weight"] = range(len(final_data))

	# Create the clusters
clust_01 = [12, 15, 11, 20, 16, 21]
clust_02 = [33, 52, 25, 30, 48, 41, 39, 50, 44, 49]
clust_03 = [43, 53, 46, 47, 54]
clust_04 = [18, 26, 37, 40]
clust_05 = [56, 55, 45, 51]
clust_06 = [14, 29, 17, 22, 19, 23, 27, 28, 38, 42, 36, 24, 32]
clust_07 = [35, 13, 31, 34]
clust_08 = [4, 7, 3, 5, 8]
clust_09 = [0]
clust_10 = [1, 2, 9, 6, 10]

	#Run the labelling function over each comment
for i in range(len(final_data)):
    label = labelling(final_data["comms"][i])
    final_data["topic_nb"][i] = label[0]
#     final_data["weight"][i] = label[1]
    

	#Apply the true label
final_data["topic_label"] = range(len(final_data))

for i in range(len(final_data)):
    if final_data["topic_nb"][i] in clust_01:
        final_data["topic_label"][i] = Topic1

    if final_data["topic_nb"][i] in clust_02:
        final_data["topic_label"][i] = Topic2
        
    if final_data["topic_nb"][i] in clust_03:
        final_data["topic_label"][i] = Topic3
    
    if final_data["topic_nb"][i] in clust_04:
        final_data["topic_label"][i] = Topic4
    
    if final_data["topic_nb"][i] in clust_05:
        final_data["topic_label"][i] = Topic5
    
    if final_data["topic_nb"][i] in clust_06:
        final_data["topic_label"][i] = Topic6
    
    if final_data["topic_nb"][i] in clust_07:
        final_data["topic_label"][i] = Topic7
    
    if final_data["topic_nb"][i] in clust_08:
        final_data["topic_label"][i] = Topic8
    
    if final_data["topic_nb"][i] in clust_09:
        final_data["topic_label"][i] = Topic9
    
    if final_data["topic_nb"][i] in clust_10:
        final_data["topic_label"][i] = Topic10
    
    #Progress bar
    # if i == 50290:
        # print("25%")

    #if i == 100580:
       # print("50%")

    # if i == 150870:
        # print("75%")

## Save the results
final_data.to_csv('Final_data.csv', index=False)
  