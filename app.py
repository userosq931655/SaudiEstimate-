from flask import Flask, render_template, url_for, request, jsonify
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, StringField
import joblib
import pandas as pd
import numpy as np
from statistics import stdev

# from app import VC

app = Flask(__name__)
app.config['SECRET_KEY'] = 'HostTic'

df = pd.read_csv('BigDataV5.csv')

city = df.city.unique()
city = city.tolist()
city.insert(0,'--Select One--')

color = df.color.unique()
color = color.tolist()
color.insert(0,'--Select One--')

make2 = df.make.unique()
make2 = make2.tolist()

trans = ["--Select One--", "Automatic", "Manual"]
# dic to make the keys as make2
all = dict.fromkeys(make2)
list = []
for i in make2:
    x = df.query('make== @i ')['model'].unique()
    x = x.tolist()
    all[i] = x
# group by brand ^^

class VC(FlaskForm):
    year = [x for x in range(2020,1999,-1)]
    year.insert(0, '--Select One--')

    make2.insert(0,'--Select One--')

    make = SelectField('make', choices=make2)
    modelcar = SelectField('modelcar')
    year = SelectField('year', choices=year)
    trans = SelectField('trans',choices=trans)
    color = SelectField('color', choices=color)
    city = SelectField('city', choices=city)
    mileage = IntegerField('mileage')
    country = SelectField('country')
    location = SelectField('location')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/valuation', methods=["GET", "POST"])
def valuation():
    form = VC()
    predicted_value = None
    low_vlaue = None
    high_vlaue = None
    data = None
    make = None
    modelcar = None
    year = None
    trans = None
    color = None
    city = None
    mileage = None
    if request.method=="POST" :
        make = form.make.data
        modelcar = form.modelcar.data
        year = int(form.year.data)
        trans = form.trans.data
        color = form.color.data
        city = form.city.data
        mileage = int(form.mileage.data)



        print(make,modelcar,year,trans,color,city,mileage)

        # Load model
        model = joblib.load('vehicle.pkl')

        # Load data set
        encoder = joblib.load('encoder.pkl')

        # Car feature parameters for prediction
        data = [{'brand': make, 'model': modelcar, 'year': year, 'mileage': mileage,
                 'city': city, 'color': color, 'transmission': trans}]
        df = pd.DataFrame(data)
        print(data)

        # data2 = [{'brand': 'Land Rover', 'model': 'Range Rover Sport Supercharged', 'year': 2014, 'mileage': 100000,
        #          'city': 'Jeddah', 'color': 'White', 'transmission': 'Automatic'}]
        # print(data2)
        # print(data==data2)

        car_to_value = encoder.transform(df)
        car_to_value.to_numpy()

        # Run the model and make a prediction for car in car_to_value array
        predicted_car_values = model.predict(car_to_value.to_numpy())

        # Only first car prediction returned from array
        predicted_value = predicted_car_values[0]
        predicted_value = int(predicted_value)


        #RANGE
        estimations = []
        for i, x in enumerate(model.estimators_):
            estimations.append(model.estimators_[i].predict(car_to_value.to_numpy()))
        arr = np.array(estimations)
        rng_std = stdev(arr.ravel())/2

        low_vlaue = predicted_value-rng_std
        high_vlaue = predicted_value+rng_std
        low_vlaue = int(low_vlaue)
        high_vlaue = int(high_vlaue)
        print("This vehicle has an estimated value of SR{:,.2f}".format(predicted_value))




    return render_template("valuation.html",form=form , result=predicted_value , cardetails=data , low_vlaue=low_vlaue,high_vlaue=high_vlaue ,make=make ,modelcar=modelcar ,
                           year=year,color=color , trans= trans , mileage = mileage ,city=city, dict=all)

@app.route("/get_model/<make>", methods=["POST","GET"])
def get_model(make):
    form = VC()
    result = all[make]
    print(result)

    return jsonify({"x": result})

@app.route('/contact-us')
def contactUS():
    return render_template("contact-us.html")


if __name__ == '__main__':
    app.run(debug=True)

