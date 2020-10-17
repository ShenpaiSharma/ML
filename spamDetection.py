import nltk
nltk.download_shell

messages = [line.rstrip() for line in open('SMSSpamCollection')]

print(len(messages))

messages[5]

messages[0]

for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
messages = pd.read_csv('SMSSpamCollection', sep = '\t', names = ['label', 'message'])

messages.head()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)

messages['length'].plot.hist(bins=35)

messages['length'].max()

import string
mess = 'Sample message! Notice: it has punctuation.'
string.punctuation

nopunc = [c for c in mess if c not in string.punctuation]


from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('english')

#nopunc = ''.join(nopunc)

#clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords('english')]


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages['message'].head().apply(text_process)


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))
mess4 = messages['message'][3]
bow4 = bow_transformer.transform([mess4])
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of sparse Matrix:',messages_bow.shape)

from sklearn.feature_extraction.text import TfidfTransformer 
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['University']]

message_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(message_tfidf,messages['label'])
spam_detect_model.predict(tfidf4)[0]

all_pred = spam_detect_model.predict(message_tfidf)


from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size = 0.3)


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer = text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
        ])

pipeline.fit(msg_train,label_train)    
predict = pipeline.predict(msg_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(label_test,predict))
