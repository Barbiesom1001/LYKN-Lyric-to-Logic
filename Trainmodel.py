import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    'day_since_release': [1014, 924, 833, 742, 680, 588, 478, 30],
    'song_length': [215, 198, 210, 205, 225, 192, 185, 190],
    'member_count': [5, 5, 5, 5, 7, 5, 5, 5],
    'trend_score': [4, 3, 4, 3, 5, 2, 2, 4],
    'views_millions': [4.8, 2.5, 4.1, 1.8, 15.0, 1.2, 0.9, 0.5]
}

df = pd.DataFrame(data)

x = df[['day_since_release', 'song_length', 'member_count', 'trend_score']]
y = df['views_millions']

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, 'LYKN_model.pkl')
print("Model Trainded and saved")