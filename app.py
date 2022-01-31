import flask
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template
from io import BytesIO
import base64
from flask import url_for,request
import os
df = pd.read_csv("FuelConsumption.csv", encoding= 'unicode_escape')
df.replace([np.inf, -np.inf], np.NaN, inplace=True)
df.fillna(999, inplace=True)
df.head()

cdf = df[['Engine_Size','Cylinders','Fuel_Consumption_city','Fuel_consumption_Hwy','Fuel_consumption_Comb','CO2_Emissions','Smog ']]
cdf.head(9)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
matplotlib.use('Agg')

model=pickle.load(open('ensemble_model.pkl','rb'))
model2=pickle.load(open('xgboost_random_model.pkl','rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/about', methods=['GET'])
def display_home_page():
	return (flask.render_template('about.html',title="about"))
@app.route('/carbon', methods=['GET'])
def display_carbon_neutral_page():
	return (flask.render_template('carbon.html',title="carbon"))
@app.route('/aqi', methods=['GET'])
def display_aqi():
    return render_template('index.html')
@app.route('/co2', methods=['GET'])
def display_CO2():
    return render_template('indexco2.html')
@app.route('/result2', methods=['POST'])
def predict1():
    if request.method == 'POST':
        avg_temp = float(request.form['Average_temp'])
        max_temp = float(request.form['Max_temp'])
        min_temp = float(request.form['Min_temp'])
        at_pres = float(request.form['Atmospheric_pressure'])
        avg_hum = float(request.form['Average_humidity'])
        avg_vis = float(request.form['Average_visibility'])
        avg_speed = float(request.form['Average_windspeed'])
        max_sustained = float(request.form['Max sustained wind speed'])
        
        data = np.array([[avg_temp,max_temp, min_temp, at_pres, avg_hum, avg_vis, avg_speed, max_sustained]])
         
        my_prediction = model2.predict(data)
        my_prediction =np.round(my_prediction,2) 
        if (my_prediction>=0 and my_prediction<=30):
            inference="Good"
        elif(my_prediction>=31 and my_prediction<=60):
            inference="Satisfactory"
        elif(my_prediction>=61 and my_prediction<=100):
            inference="Moderately Polluted"
        elif(my_prediction>=101 and my_prediction<=170.4):
            inference="Unhealthy"  
        elif(my_prediction>=170.5 and my_prediction<=220.4):
            inference="Very Unhealthy"
        elif(my_prediction>=220.5):
          inference="Hazardous"
        
        return render_template('result2.html', prediction=my_prediction,inf=inference)

@app.route("/result", methods=['POST'])
def predict():

    if request.method == 'POST':
        ENGINESIZE = float(request.form['ENGINESIZE'])
        CYLINDERS = int(request.form['CYLINDERS'])
        FUELCONSUMPTION_CITY =float(request.form['FUELCONSUMPTION_CITY'])
        FUELCONSUMPTION_HWY = float(request.form['FUELCONSUMPTION_HWY'])
        
        input_variables = pd.DataFrame([[ENGINESIZE,CYLINDERS,FUELCONSUMPTION_CITY,FUELCONSUMPTION_HWY]],
                                       columns=['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY'],
                                       dtype=float,
                                       index=['input'])
        prediction = model.predict([[ENGINESIZE ,CYLINDERS,FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY]])
        prediction = prediction[0]
        prediction2=model.predict(input_variables)[0]
        final = (str(round(prediction))) + " g/Km" 
        final2 = (str(round(prediction2))) 
        if(round(prediction)>200):
            inference="high"
        elif(round(prediction)>150 and round(prediction)<200):
            inference="moderate"
        else:
            inference="low"
    return render_template('result.html', result=final, inf=inference)

@app.route('/plot2')
def plot2():
        img = BytesIO()
        data=pd.read_csv('Carbon Dioxide Emission.csv')
        df=pd.DataFrame (data)
        year = df ['Decimal_Date'].tail(400)
        avg = df ['Average'].tail(400)
        x=year
        y=avg
        plt.plot (x,y, label='CO2 Emission Rate', color='#8A3324')
        plt.title ("\nCO2 Emission Rates over the years\n", color='k')
        plt.xlabel ("\nYear\n", color= 'k')
        plt.ylabel ("\nCO2 Emission  (PPM)\n", color= 'k')
        plt.legend ()
        plt.grid (True, color='grey')
        plt.savefig(img, format='png', dpi=100)
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('plot2.html', plot_url=plot_url)  

@app.route('/plot3')
def plot3():
        img = BytesIO()
        data=pd.read_csv('Land Ocean Temp Index.csv')
        df=pd.DataFrame (data)
        year= df ['Year']
        temp_change_rate= df ['No_Smoothing']
        x=year
        y=temp_change_rate
        plt.plot (x,y, label='Change in Temperature', color='#1a0000')
        plt.title ("\nChange In Global Surface Temperature\n", color='k')
        plt.xlabel ("\nYear\n", color= 'k')
        plt.ylabel ("\nChange in Temperature\n", color= 'k')
        plt.legend ()
        plt.grid (True, color='gray')
        
        plt.savefig(img, format='png' , dpi=100)
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('plot3.html', plot_url=plot_url)  


@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html',title="predict"))

if __name__ == '__main__':
    app.debug = True
    app.run()
