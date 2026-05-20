from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model = Sequential()
model.compile()
model.fit(x_train, y_train)
model.evaluate(x_test,y_test)

