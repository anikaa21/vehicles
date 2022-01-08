import flask
import pickle
import pandas as pd
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template
from io import BytesIO
import base64


df = pd.read_csv("FuelConsumption.csv", encoding= 'unicode_escape')
df.replace([np.inf, -np.inf], np.NaN, inplace=True)
df.fillna(999, inplace=True)
df.head()

cdf = df[['Engine_Size','Cylinders','Fuel_Consumption_city ','Fuel_consumption_Hwy','Fuel_consumption_Comb','CO2_Emissions','Smog ']]
cdf.head(9)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
matplotlib.use('Agg')

# Use pickle to load in the pre-trained model
model=pickle.load(open('ensemble_model.pkl','rb'))
model2=pickle.load(open('ensemble_model2.pkl','rb'))
# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')
@app.route('/about', methods=['GET'])
def display_home_page():
	return (flask.render_template('about.html',title="about"))
@app.route('/risk', methods=['GET'])
def display_risk_page():
	return (flask.render_template('risk.html',title="risk"))
@app.route('/carbon', methods=['GET'])
def display_carbon_neutral_page():
	return (flask.render_template('carbon.html',title="carbon"))

@app.route('/plot')
def plot():
        img = BytesIO()
        plt.figure(figsize=(9,9))

        sns.barplot(x='Cylinders', y='CO2_Emissions',data = df)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('plot.html', plot_url=plot_url)  

@app.route('/plot2')
def plot2():
        img = BytesIO()
        plt.figure(figsize=(9,9))
        sns.barplot(x = 'Engine_Size', y='CO2_Emissions', data =df[:25])
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('plot2.html', plot_url=plot_url)  

@app.route('/plot3')
def plot3():
        img = BytesIO()
        plt.figure(figsize=(9,9))
        sns.barplot(x = 'Fuel_consumption_Comb', y='CO2_Emissions',data =df[:25])
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('plot3.html', plot_url=plot_url)  
         
# Set up the main route

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html',title="predict"))
    
    if flask.request.method == 'POST':
        # Extract the input
        ENGINESIZE = flask.request.form['ENGINESIZE']
        CYLINDERS = flask.request.form['CYLINDERS']
        FUELCONSUMPTION_CITY = flask.request.form['FUELCONSUMPTION_CITY']
        FUELCONSUMPTION_HWY = flask.request.form['FUELCONSUMPTION_HWY']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[ENGINESIZE,CYLINDERS,FUELCONSUMPTION_CITY,FUELCONSUMPTION_HWY]],
                                       columns=['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY'],
                                       dtype=float,
                                       index=['input'])
        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
        prediction2=model2.predict(input_variables)[0]

        final = (str(round(prediction))) + " g/Km" 
        final2 = (str(round(prediction2))) 
        if(round(prediction)>200):
            inference="high"
        elif(round(prediction)>150 and round(prediction)<200):
            inference="average"
        else:
            inference="low"

        if(round(prediction2)>7):
            inference2="high"
        elif(round(prediction2)>5 and round(prediction2)<7):
            inference2="average"
        else:
            inference2="low"
        
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'ENGINESIZE':ENGINESIZE,
                                                     'CYLINDERS':CYLINDERS,
                                                     'FUELCONSUMPTION_CITY':FUELCONSUMPTION_CITY,
                                                     'FUELCONSUMPTION_HWY':FUELCONSUMPTION_HWY},
                                     result=final ,result2=final2 ,inf=inference,inf2=inference2)
                                     
# Set up the main route
if __name__ == '__main__':
    app.debug = True
    app.run()
