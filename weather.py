
from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

# Weather API Configuration
BASE_URL = "http://api.openweathermap.org/data/2.5/"
API_KEY = "bf9fcfa3dc51c0fc32ac0f7200b5d29f"  # Replace with a valid API key

def fetch_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None

        return {
            'city': data.get('name', 'N/A'),
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'] if 'weather' in data else 'N/A',
            'country': data['sys'].get('country', 'N/A'),
            'wind_gust_dir': data['wind'].get('deg', 0),
            'pressure': data['main'].get('pressure', 0),
            'wind_gust_speed': data['wind'].get('speed', 0),
        }
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except KeyError as e:
        print(f"Missing expected key in response: {e}")
        return None

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    wind_dir_encoder = LabelEncoder()
    rain_encoder = LabelEncoder()
    
    data['WindGustDir'] = wind_dir_encoder.fit_transform(data['WindGustDir'].astype(str))
    data['RainTomorrow'] = rain_encoder.fit_transform(data['RainTomorrow'].astype(str))

    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    Y = data['RainTomorrow']

    return X, Y, wind_dir_encoder

def prepare_regression_data(data, feature):
    X, Y = [], []
    
    for i in range(len(data)-1):
        X.append(data[feature].iloc[i])
        Y.append(data[feature].iloc[i+1])

    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)
    
    return X, Y

def train_rain_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    
    return model, mse

def train_regression_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, Y_train)
    
    return model

def predict_future(model, current_value):
    predictions = []
    next_value = current_value
    
    for _ in range(5):
        next_value = model.predict(np.array([[next_value]]))[0]
        predictions.append(next_value)
    
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    weather_data = None
    error = None

    if request.method == 'POST':
        city = request.form.get('city')
        try:
            current_weather = fetch_current_weather(city)
            if not current_weather:
                error = "Failed to fetch weather data."
                return render_template('index.html', error=error)

            historical_data = read_historical_data('data/weather.csv')
            if historical_data.empty:
                error = "Historical data is empty."
                return render_template('index.html', error=error)

            X, Y, wind_dir_encoder = prepare_data(historical_data)
            rain_model, mse = train_rain_model(X, Y)

            wind_deg = current_weather['wind_gust_dir'] % 360
            compass_points = [
                ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
                ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
                ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
                ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
                ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
                ("NNW", 326.25, 348.75), ("N", 348.75, 360)
            ]
            compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "N")

            if compass_direction not in wind_dir_encoder.classes_:
                compass_direction = wind_dir_encoder.classes_[0]

            compass_encoded = wind_dir_encoder.transform([compass_direction])[0]

            current_df = pd.DataFrame([{
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'WindGustDir': compass_encoded,
                'WindGustSpeed': current_weather['wind_gust_speed'],
                'Humidity': current_weather['humidity'],
                'Pressure': current_weather['pressure'],
                'Temp': current_weather['current_temp'],
            }], columns=X.columns)

            rain_prediction = rain_model.predict(current_df)[0]

            X_temp, Y_temp = prepare_regression_data(historical_data, 'Temp')
            X_humidity, Y_humidity = prepare_regression_data(historical_data, 'Humidity')

            temp_model = train_regression_model(X_temp, Y_temp)
            humidity_model = train_regression_model(X_humidity, Y_humidity)

            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_humidity = predict_future(humidity_model, current_weather['humidity'])

            timezone = pytz.timezone('Asia/Kolkata')
            current_time = datetime.now(timezone)
            future_times = [(current_time + timedelta(hours=i+1)).strftime("%H:%M") for i in range(5)]

            weather_data = {
                'city': city,
                'country': current_weather['country'],
                'current_temp': current_weather['current_temp'],
                'feels_like': current_weather['feels_like'],
                'temp_min': current_weather['temp_min'],
                'temp_max': current_weather['temp_max'],
                'humidity': current_weather['humidity'],
                'description': current_weather['description'],
                'rain_prediction': 'Yes' if rain_prediction else 'No',
                'future_temp': list(zip(future_times, map(lambda x: round(x, 1), future_temp))),
                'future_humidity': list(zip(future_times, map(lambda x: round(x, 1), future_humidity))),
            }
        except Exception as e:
            error = str(e)

    return render_template('index.html', weather=weather_data, error=error)

if __name__ == '__main__':
    app.run(debug=True)

