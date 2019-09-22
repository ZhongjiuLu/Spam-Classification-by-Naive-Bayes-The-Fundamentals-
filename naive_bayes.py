"""
Build a Naive Bayes spam filter
@author: Zhongjiu Lu
"""
# import 
import numpy as np
import pandas as pd
import re

""" 
load the data into a Python data frame
"""
df = pd.read_table('SMSSpamCollection.txt', delimiter="\t", names=('label', 'sms'))

""" 
Pre-process the SMS messages: Remove all punctuation and numbers from the SMS
messages, and change all messages to lower case. 
"""

df['length'] = df.sms.apply(lambda x: len(x))
    
whitelist = set ('abcdefghijklmnopqrstuvwxy ')
df['sms'] = df.sms.apply(lambda x: ''.join(filter(whitelist.__contains__, x.lower())))

"""
Shuffle the messages and split them into a training set (2,500 messages), a validation
set (1,000 messages) and a test set (all remaining messages).
"""
# shuffle without pandas
df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
# splitting
msgs = list(df['sms'])
lbls =list(df['label'])
trainingMsgs = msgs[:2500]
trainingLbls = lbls[:2500]
valMsgs = msgs[2500:3500]
valLbls=lbls[2500:3500]
testingMsgs = msgs[3500:]
testingLbls=lbls[3500:]

"""
Although Python's SciKit-Learn library has a Naive Bayes classier, 
it works with continuous probability distributions and assumes numerical features. 
Although it is possible to transform categorical variables into numerical features using a binary encoding, 
we will instead build a simple Naive Bayes classier from scratch:    
"""    

"""
The functions 'train' and 'train2' calculate and store the prior probabilities and likelihoods.
In Naive Bayes this is all the training phase does. The 'predict' function repeatedly applies
Bayes' Theorem to every word in the constructed dictionary, and based on the posterior
probability it classi
es the message as 'spam' or 'ham'. The 'score' function calls 'predict'


for multiple messages and compares the outcomes with the supplied 'ground truth' labels
and thus evaluates the classi
er. It also computes and returns a confusion matrix.
The difference between 'train' and 'train2' is that the latter function only takes into
account words that are 20 times more likely to appear in a spam message than a ham
message. Not all words are equally informative. Using 'train2' will decrease the size of the
dictionary signi
cantly, thus making the classi
er more ecient. There are multiple other
ideas that one can use to construct more informative dictionaries. For example, you can
treat words with the same root (`go', 'goes', 'went', 'gone', 'going') as the same word.

"""

class NaiveBayesForSpam:
    #ham messages are the messages of ham emails
    #spam messages are the messages of spam emails
    def train (self, hamMessages, spamMessages):
        #return a set of letters of combined ham and spam messages
        self.words = set (' '.join (hamMessages + spamMessages).split())
        #initialize an array
        self.priors = np.zeros(2)
        #prior probability of P(ham emails)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len (spamMessages))
        #prior probability of P(Spam emails)
        self.priors[1] = 1.0 - self.priors[0]
        #initialize the likelihood list        
        self.likelihoods = []
        # for each word in the messages, we calculate its likelihood(slide 21)
        for i, w in enumerate (self.words):
            # len ([m for m in hamMessages if w in m]) shows how many times a word appear in messages from ham emails.
            # Laplace correction: add 1 to every entry in the frequency matrix to eliminate the situation that 
            # the word doesn't show up given the email is ham or the word doesn't show up given the email is spam
            # len(hamMessages) show the number of ham emails
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len (hamMessages)
            # prob2 shows the likelihood for spam email
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            # boundary for likelihood at 0.95
            self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
        #to select both prob1 and prob2 more easily
        #self.likelihoods[:,1] is the list of both prob1 and prob2,
        self.likelihoods = np.array (self.likelihoods).T
        len(self.likelihoods)
        
    def train2 (self, hamMessages, spamMessages):
        self.words = set (' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros (2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len (spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        spamkeywords = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len (hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            # a word is appear in spam email 20 times more frequent than in ham emails is a spam key word
            # the likelihood will only be accounted if difference is large enough
            if prob1 * 20 < prob2:
                self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
                spamkeywords.append (w)
        self.words = spamkeywords
        self.likelihoods = np.array (self.likelihoods).T
        len(self.likelihoods)

    def predict (self, message):
        #if we apply train, we use priors that is defined in train
        #if we apply train2, we use priors that is defined in train2
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower():  # convert to lower-case
                # posteriors[0] = P(word n appear | ham email )*P(word n+1 appear | ham email )
                # posteriors[1] = P(word n appear | spam email )*P(word n+1 appear | spam email )
                posteriors *= self.likelihoods[:,i]
            else:                                   
                # posteriors[0] = P(no word n | ham email )*P(no word n+1 | ham email )
                # posteriors[1] = P(no word n | spam email )*P(no word n+1 | spam email ) 
                posteriors *= np.ones (2) - self.likelihoods[:,i]
            #np.linalg.norm (posteriors , ord =1) = max(abs(posteriors[0])+abs(posteriors[1]),0)
            posteriors = posteriors / np.linalg.norm (posteriors, ord =1)  # normalise
        #posteriors[0]is greater than posteriors[1] relatively
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]    

    def score (self, messages, labels):
        confusion = np.zeros(4).reshape (2,2)
        for m, l in zip (messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion
    
    
    
"""
use the training set to train the classi
ers 'train' and 'train2'.
"""
hammsgs = [m for(m,l) in zip (trainingMsgs, trainingLbls) if 'ham' in l]
spammsgs = [m for(m, l) in zip (trainingMsgs, trainingLbls) if 'spam' in l]


clf = NaiveBayesForSpam()
clf.train(hammsgs , spammsgs)

"""
Using the validation set to evaluate how the two classi
ers performs out of sample.
"""
# model train
score, confusion = clf.score(valMsgs, valLbls)
#print(score)
#print(confusion)

"""
For 'train' function, the confusion matrix is:
    
[[872.  12.]
 [ 14. 102.]]

with an overall performance of 0.974.
"""

# model train2
clf = NaiveBayesForSpam()
clf.train2(hammsgs, spammsgs)
score, confusion = clf.score(valMsgs, valLbls)
#print(score)
#print(confusion)
"""
For 'train2' function, the confusion matrix is:
    
[[881.  21.]
 [  5.  93.]]

with an overall performance of 0.974
"""

"""
Since the data is not equally divided into the two classes. 
As a baseline, let's see what the success rate would be if we always guessed `ham':
"""

# print ('base score', len ([ 1 for l in valLbls if 'ham' in l])/float(len(valLbls)))

"""
This gives a performance of 0.886, which is inferior. Let us also calculate our in sample error
"""

"""
'train2' classi
er is faster than 'train' 
and yields a better accuracy both on the training and the alidation set.

Here are the reasons:
    
    
    
    The di
erence between 'train' and 'train2' is that the latter function only 
    takes into account words that are 20 times more likely to appear in a spam 
    message than a ham message. This makes the dictionary much smaller 
    and hence speeds up the algorithm. However, why would a simpler model 
    (less words) give a better performance, not just out of sample 
    (in the validation set), where over
tting may deteriorate the quality of more
    sophisticated methods, but also in sample? The reason is the conditional 
    independence assumption of Naive Bayes. With more words in our dictionary, 
    the number of correlated words is larger and therefore the violation of 
    our theoretical assumption stronger. It appears that 'train2' mediates this 
    effect.

"""

"""
reduce false positives at the expense of possibly having more false negatives 
(spam messages classi
ed as ham messages):
change the threshold like the following
    
if (posteriors[0] > 0.5):
    return (['ham',posteriors[0]])
    
if (posteriors[0] > 0.2):
    return (['ham',posteriors[0]])
    
"""





















