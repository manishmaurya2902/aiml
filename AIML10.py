from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
    'Area': [1000, 1500, 1800, 2400],
    'Bedrooms': [2, 3, 3, 4],
    'Bathrooms': [2, 2, 3, 3],
    'price': [3500000, 5000000, 6500000, 9000000]
}

df = pd.DataFrame(data)

x = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(x, y)

prediction = model.predict(
    pd.DataFrame([[2000, 3, 2]],
    columns=['Area', 'Bedrooms', 'Bathrooms'])
)

print("Prediction Price:", prediction[0])