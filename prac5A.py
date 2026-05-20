from sklearn import svm

# svm - orange, apple

x = [[150,9],[140,8],[120,6],[125,7]]
y = ["apple","apple","orange","orange"]

model = svm.SVC() #linear boundry(decided by kernel)

model.fit(x,y)

prediction = model.predict([[170,10]])

print("prediction is: ", prediction)
