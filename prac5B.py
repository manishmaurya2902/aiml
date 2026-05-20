from sklearn.tree import DecisionTreeClassifier

x = [[30,35000],[25,20000],[50,50000],[20,100000]]
y = ['yes', 'no', 'yes', 'no']

model = DecisionTreeClassifier()
model.fit(x,y)

print("prediction is: ",model.predict([[24,60000]]))
