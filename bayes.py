from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

newsgroups_train = datasets.fetch_20newsgroups(subset='train')
newsgroups_test = datasets.fetch_20newsgroups(subset='test')

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(newsgroups_train)
x_test = vectorizer.transform(newsgroups_test)

y_train = newsgroups_train.target
y_test = newsgroups_test.target

model = MultinomialNB()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print (model.score(x_test, y_test))
print(metrics.classification_report(y_test, predictions))
