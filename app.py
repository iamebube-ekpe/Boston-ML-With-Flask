import pandas as pd
from flask import Flask, render_template, request
import pickle

## Using pickle to load the trained model
with open(f'model/boston_model.pkl', 'rb') as f:
   model = pickle.load(f)

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method =='GET':
      return(render_template('index.html'))

   if request.method == 'POST':
      rm = request.form['rm']
      zn = request.form['zn']
      b = request.form['b']

      input_variables = pd.DataFrame([[rm, zn, b]], columns=['RM', 'ZN', 'B'], dtype=float)
      prediction = model.predict(input_variables)[0]

      return(render_template('index.html', original_input={'RM':rm, 'ZN':zn, 'B':b},result=prediction))


if __name__ == '__main__':
    app.run()
