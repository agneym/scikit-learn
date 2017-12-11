from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz

breast_cancer = datasets.load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=13)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

graph_data = tree.export_graphviz(model, out_file=None, feature_names=breast_cancer.feature_names, filled=True)
graph = graphviz.Source(graph_data)
graph.render("breast_cancer", view=True)

print(predictions)
print (model.score(x_test, y_test))
print(metrics.classification_report(y_test, predictions))
