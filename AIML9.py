from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X = data.data
y = data.target

model = KNeighborsClassifier()

param_grid = {'n_neighbors': [3,5,7,9]}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X, y)

print("Best parameters: ", grid.best_params_)
print("Best Accuracy: ", grid.best_score_)