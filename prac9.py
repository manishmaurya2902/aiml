from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

data = load_iris()

x = data.data
y = data.target

model = KNeighborsClassifier()

grid = GridSearchCV(model, {'n_neighbors':[3,5,7,9]})

grid.fit(x,y)

print("best parameter: ", grid.best_params_)
print("best score: ",grid.best_score_)