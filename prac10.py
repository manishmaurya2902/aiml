import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'area':[1000,2000,3000,4000],
    'bedroom':[2,3,4,4],
    'washroom':[3,3,4,4],
    'price':[3000000,4000000,5000000,6000000]
}

table = pd.DataFrame(data)
x = table[['area','bedroom','washroom']]
y = table['price']

model = LinearRegression()
model.fit(x,y)

prediction = model.predict(
    pd.DataFrame([[5000,4,5]], columns=['area','bedroom','washroom'])
)

print("predicted price: ", prediction)