from flask import Flask, render_template, request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    exp1=float(request.values['SL'])
    exp1=np.reshape(exp1,(-1,1))
    exp2=float(request.values['SW'])
    exp2=np.reshape(exp2,(-1,1))
    exp3=float(request.values['PL'])
    exp3=np.reshape(exp3,(-1,1))
    exp=[exp1,exp2,exp3]
    exp=np.reshape(exp,(-1,3))
    output=model.predict(exp)
    output=output.item()
    return render_template('res.html',prediction_text="Your iris is classified as {} ".format(output))

if __name__=='__main__':
    app.run(debug=True)