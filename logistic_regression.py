from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=13)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

#accuracy score
print(model.score(x_test, y_test))

print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))

