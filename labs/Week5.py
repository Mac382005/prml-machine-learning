##########################################################################
## Classification of news articles into different topics using Naïve Bayes models
##########################################################################

# Introduction2
# Text Classification is one of the widely use from a task of natural language processing (NLP).
# It is also one of the important and typical task in pattern recognition. Text classification
# aims to automatically classify text documents into one or more defined categories
# which can be a web page, library book, media articles, gallery etc. Some examples
# of text classification can be:

# - Understanding audience sentiment from social media,
# - Detection of spam and non-spam emails,
# - Auto tagging of customer queries, and
# - Categorization of news articles into defined topics.

# In this lab, we would like to demonstrate how we can categorize different new articles
# into defined topics using sklearn library.

###########################################################################
## Dataset
# The dataset used is the 20 Newsgroup dataset (http://qwone.com/~jason/20Newsgroups/).
# The 20 Newsgroups data set is a collection of about 20000 newsgroup documents, partitioned
# (nearly) evenly across 20 different newsgroups. The dataset was originally collected by
# Ken Lang. The 20 newsgroups dataset has become a popular data for experiments in text
# applications of machine learning techniques including text classification and text clustering.

# The dataset used in this lab is extracted from the 20 Newsgroups dataset sorted by date
# in which duplicates and some headers removed. The dataset contains 18846 documents.
# The data is categorized into 20 different newsgroups, each corresponding to a different
# topic. Some of the newsgroups are very closely related to each other
# (e.g. comp.sys.ibm.pc.hardware / comp.sys.mac.hardware), while others are highly
# unrelated (e.g misc.forsale / soc.religion.christian). Here is a list of the 20 newsgroups,
# partitioned (more or less) according to subject matter:

# comp.graphics
# comp.os.ms-windows.misc
# comp.sys.ibm.pc.hardware
# comp.sys.mac.hardware
# comp.windows.x
# rec.autos
# rec.motorcycles
# rec.sport.baseball
# rec.sport.hockey
# sci.crypt
# sci.electronics
# sci.med
# sci.space
# misc.forsale
# talk.politics.misc
# talk.politics.guns
# talk.politics.mideast
# talk.religion.misc
# alt.atheism
# soc.religion.christian
###########################################################################


## Step 1: Loading data
from sklearn.datasets import fetch_20newsgroups

print("Loading data...")
newsgroups = fetch_20newsgroups(subset='all')

# Loading the whole dataset takes some time, so you can load partial data for exploration purpose using the following two lines
#categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
#newsgroups = fetch_20newsgroups(subset='all', categories=categories)

print("Data loaded!")

## Step 2: Exploring the dataset
newsgroups.data[0]
list(newsgroups.target_names)

newsgroups.filenames.shape
newsgroups.target.shape

## Step 3: Feature extraction (using Bag of Words)
# This step will calculate a bag of words and their frequencies.
# We use the CountVectorizer from sklearn to do this.

# import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Let's see an example first
example_document = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]


cv = CountVectorizer()
cv.fit_transform(example_document)
cv.vocabulary_

cv.fit_transform(example_document).toarray()

# Output a bag of words and their frequencey
word_list = cv.get_feature_names_out()
count_list = cv.fit_transform(example_document).toarray().sum(axis=0)

print(dict(zip(word_list,count_list)))

# Feature Extraction for the 20NewsGroup dataset
features = cv.fit_transform(newsgroups.data)
features.shape
features.toarray()


## Step 4: Split data into trainning and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    newsgroups.target,
                                                    train_size=0.8,
                                                    random_state=0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape


## Step 5: Build and Train a Multinomial Naïve Bayes model
from sklearn.naive_bayes import MultinomialNB

# build the model
model = MultinomialNB(alpha=1)
# train the model
model.fit(X_train,y_train)


## Step 6: Make predictions
y_pred = model.predict(X_test)

print(y_test)
print(y_pred)


## Step 7: Evaluate the model
from sklearn import metrics
from sklearn.metrics import classification_report

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))

# Classification report
report = classification_report(y_test, y_pred, target_names = newsgroups.target_names)
print(report)


from sklearn.metrics import confusion_matrix

# Confusion matrix
confusion_matrix(y_test, y_pred)


## Step 8: Tuning - finding the best parameter alpha for the model using GridSearchCV
# GridSearchCV is a method in scikit-learn (Python) that helps you find the best
# hyperparameters for a machine learning model.

# Grid Search:
# - You define a set (a “grid”) of possible hyperparameter values.
# - The algorithm trains and evaluates the model on every possible combination of these values.
# CV (Cross-Validation):
# - Instead of just splitting data once into train/test, it uses k-fold cross-validation to make sure results are reliable.
# - The model is trained on k-1 folds and validated on the remaining fold, repeated k times.
# Finally, it reports the best parameters and the corresponding performance.

#Import the Grid Search
from sklearn.model_selection import GridSearchCV

model = MultinomialNB()
# Parameters setting for alpha = 10, 1, 0.1, 0.001, 0.0001
params = {'alpha': [10, 1, 1e-1, 1e-2, 1e-3]}

# Running the Grid Search (5-fold cv) on parameters and fit the trainning data
grs = GridSearchCV(model, param_grid=params, cv = 5)
grs.fit(X_train, y_train)

# Output the optimal values
print("Best Hyper Parameters:",grs.best_params_)

# Make a prediction and calculate metrics
y_pred = grs.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))