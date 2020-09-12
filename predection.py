import joblib
import pandas as pd
# from app import VC

# dataIN = VC()

# Load model
model = joblib.load('vehicle.pkl')

# Load data set
encoder = joblib.load('encoder.pkl')

# Car feature parameters for prediction
data = [{'brand': 'Land Rover', 'model': 'Range Rover Sport Supercharged', 'year': 2014, 'mileage': 100000, 'city': 'Jeddah', 'color': 'White', 'transmission': 'Automatic'}]
df = pd.DataFrame(data)

car_to_value = encoder.transform(df)
car_to_value.to_numpy()

# Run the model and make a prediction for car in car_to_value array
predicted_car_values = model.predict(car_to_value.to_numpy())

# Only first car prediction returned from array
predicted_value = predicted_car_values[0]

print("This vehicle has an estimated value of SR{:,.2f}".format(predicted_value))