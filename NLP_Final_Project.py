#!/usr/bin/env python
# coding: utf-8

# ### FINAL PROJECT NLP

# ### Email Spam Detection Using Count Vectorization Character N-gram 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data= pd.read_csv("F:/Semester 3/NLP/Email spam.csv")
data.head(5)


# In[3]:


data.describe()


# In[4]:


#Provide the set of features we have 
data.columns


# In[5]:


data['spam'].value_counts()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


data.groupby(['spam']).describe()


# ### DATA VISULAIZATION

# In[9]:


sns.countplot(data.spam)


# ### Analyzing length of the dataset

# In[10]:


plt.figure(figsize=(12,8))
data[data.spam==0].text.apply(len).plot(kind='hist',alpha=0.6,bins=35,label='Non-Spam messages')
data[data.spam==1].text.apply(len).plot(kind='hist', color='red',alpha=0.6,bins=35,label='Spam messages')

plt.legend()
plt.xlabel("Message Length")
plt.show()


# ### WORD CLOUD FOR THE TEXT THAT IS NOT SPAM

# In[11]:


plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.spam == 0].text))
plt.imshow(wc , interpolation = 'bilinear')


# ### WORDCLOUD FOR TEXT THAT IS SPAM

# In[12]:


plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.spam == 1].text))
plt.imshow(wc , interpolation = 'bilinear')


# ###  TRANSFORMATION

# In[13]:


data.isnull().sum()


# In[14]:


#Check if we have duplicates values
data.duplicated().sum()


# In[15]:


# drop duplicates
data.drop_duplicates(inplace=True)


# ### Word Tokenize

# In[16]:


from nltk import word_tokenize


# In[17]:


def count_words(text):
    words = word_tokenize(text)
    return len(words)
data['count']=data['text'].apply(count_words)
data['count']


# In[18]:


data.groupby('spam')['count'].mean()


# ### Text Preprocessing

# ### Function to process the text data 
# 1. Remove Punctuation
# 2. Stop Words
# 3. Stemming

# In[19]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize


# In[20]:


get_ipython().run_cell_magic('time', '', 'def clean_str(string, reg = RegexpTokenizer(r\'[a-z]+\')):\n    # Clean a string with RegexpTokenizer\n    string = string.lower()\n    tokens = reg.tokenize(string)\n    return " ".join(tokens)\n\nprint(\'Before cleaning:\')\ndata.head()')


# In[21]:


print('After cleaning:')
data['text'] = data['text'].apply(lambda string: clean_str(string))
data.head()


# #### After cleaning the text. We will now carry out the process of Stemming to reduce infected words to their root

# In[22]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stemming (text):
    return ''.join([stemmer.stem(word) for word in text])
data['text']=data['text'].apply(stemming)
data.head()


# In[23]:


X = data.loc[:, 'text']
y = data.loc[:, 'spam']

print(f"Shape of X: {X.shape}\nshape of y: {y.shape}")


# ### Split into Training data and Test data

# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[25]:


print(f"Training Data Shape: {X_train.shape}\nTest Data Shape: {X_test.shape}")


# ### Now we will use Count Vectorizer to convert string data into Bag of Words ie Known Vocabulary

# In[26]:


def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(data.text)
corpus[:5]
from collections import Counter
counter = Counter(corpus)
most_common = counter.most_common(10)
most_common = dict(most_common)
most_common


# In[27]:


def get_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# 
# ###  UNIGRAM OR 1-GRAM:

# In[28]:


unigrams = get_ngrams(data['text'],10,1)
unigrams_df = pd.DataFrame(unigrams)
unigrams_df.columns=["Unigram", "Frequency"]
unigrams_df.head()


# In[29]:


sns.barplot(x=list(most_common.values()),y=list(most_common.keys()))


# ### BIGRAM:

# In[30]:


bigrams = get_ngrams(data['text'], 10,2)
bigrams_df = pd.DataFrame(bigrams)
bigrams_df.columns=["bigram", "Frequency"]
bigrams_df.head()


# In[31]:


plt.figure(figsize = (16,9))
most_common_bi = get_ngrams(data.text,10,2)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))


# ### TRIGRAM:

# In[32]:


Trigrams = get_ngrams(data['text'],10,3)
trigrams_df = pd.DataFrame(Trigrams)
trigrams_df.columns=["Trigram", "Frequency"]
trigrams_df.head()


# In[33]:


plt.figure(figsize = (16,9))
trigrams = get_ngrams(data.text,10,3)
trigrams = dict(trigrams)
sns.barplot(x=list(trigrams.values()),y=list(trigrams.keys()))


# ### QUADGRAM:

# In[34]:


quadgrams = get_ngrams(data['text'],10,4)
quadgrams_df = pd.DataFrame(quadgrams)
quadgrams_df.columns=["Quadgram", "Frequency"]
quadgrams_df.head()


# In[35]:


plt.figure(figsize = (16,9))
quadgrams = get_ngrams(data.text,10,4)
quadgrams = dict(quadgrams)
sns.barplot(x=list(quadgrams.values()),y=list(quadgrams.keys()))


# ### N th GRAM:

# In[36]:


ngrams = get_ngrams(data['text'],10,5)
ngrams_df = pd.DataFrame(ngrams)
ngrams_df.columns=["N gram", "Frequency"]
ngrams_df.head()


# In[37]:


plt.figure(figsize = (16,9))
ngrams = get_ngrams(data.text,10,5)
ngrams = dict(ngrams)
sns.barplot(x=list(ngrams.values()),y=list(ngrams.keys()))


# In[38]:


# vectorization of n grams
vec=CountVectorizer(ngram_range = (6, 6), max_features = 4435, stop_words='english')
dtv = vec.fit_transform(X_train).toarray()
print('No.of Tokens: ',len(vec.vocabulary_.keys()))


# In[39]:


print(f"Number of Observations: {dtv.shape[0]}\nTokens/Features: {dtv.shape[1]}")


# In[40]:


dtv[1]


# ### MODEL IMPLEMENTATION

# In[41]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.linear_model import LogisticRegression\nfrom time import perf_counter\nimport warnings\nwarnings.filterwarnings(action=\'ignore\')\nmodels = {\n    "Random Forest": {"model":RandomForestClassifier(), "perf":0},\n    "MultinomialNB": {"model":MultinomialNB(), "perf":0},\n    "Logistic Regr.": {"model":LogisticRegression(solver=\'liblinear\', penalty =\'l2\' , C = 1.0), "perf":0},\n}\n\nfor name, model in models.items():\n    start = perf_counter()\n    model[\'model\'].fit(dtv, y_train)\n    duration = perf_counter() - start\n    duration = round(duration,2)\n    model["perf"] = duration\n    print(f"{name:20} trained in {duration} sec")')


# In[42]:


test_dtv = vec.transform(X_test)
test_dtv = test_dtv.toarray()
print(f"Number of Observations: {test_dtv.shape[0]}\nTokens: {test_dtv.shape[1]}")


# ### Test Accuracy and Training Time

# In[43]:


models_accuracy = []
for name, model in models.items():
    models_accuracy.append([name, model["model"].score(test_dtv, y_test),model["perf"]])


# In[44]:


data_accuracy = pd.DataFrame(models_accuracy)
data_accuracy.columns = ['Model', 'Test Accuracy', 'Training time (sec)']
data_accuracy.sort_values(by = 'Test Accuracy', ascending = False, inplace=True)
data_accuracy.reset_index(drop = True, inplace=True)
data_accuracy


# In[45]:


plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y ='Test Accuracy', data = data_accuracy)
plt.title('Accuracy on the test set\n', fontsize = 15)
plt.ylim(0.825,1)
plt.show()


# In[46]:


plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y = 'Training time (sec)', data = data_accuracy)
plt.title('Training time for each model in sec', fontsize = 15)
plt.ylim(0,1)
plt.show()


# ### Logistic Regression

# In[47]:


get_ipython().run_cell_magic('time', '', "lr = LogisticRegression(solver='liblinear', penalty ='l2' , C = 1.0)\nlr.fit(dtv, y_train)\npred = lr.predict(test_dtv)")


# In[48]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# In[49]:


print(classification_report(y_test, pred))


# In[50]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ### Random Forest Classifier

# In[51]:


get_ipython().run_cell_magic('time', '', 'rfc = RandomForestClassifier()\nrfc.fit(dtv, y_train)\npred = rfc.predict(test_dtv)')


# In[52]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# ### Classification Report

# In[53]:


print(classification_report(y_test, pred))


# ### Confusion Matrix

# In[54]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ### Multinomial Naive Bayes

# In[55]:


get_ipython().run_cell_magic('time', '', 'mnb = MultinomialNB()\nmnb.fit(dtv, y_train)\npred = mnb.predict(test_dtv)')


# In[56]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# ### Classification Report

# In[57]:


print(classification_report(y_test, pred))


# ### Confusion Matrix

# In[58]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);

