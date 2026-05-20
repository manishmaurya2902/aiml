from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

x_true = [1,1,0,0,1,0,1,1]
x_pred = [1,1,1,0,1,0,1,0]

cm = confusion_matrix(x_true, x_pred)
print("confusion matrix")
print(cm)

print("precision: ", precision_score(x_true,x_pred))
print("recall: ",recall_score(x_true,x_pred))
print("accuracay: ",accuracy_score(x_true,x_pred))

Confusion Matrix

A confusion matrix is a table that shows:

how many predictions were correct
and where the model made mistakes.

It compares:

Actual values
Predicted values




