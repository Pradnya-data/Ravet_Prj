from flask import Flask,render_template,request
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris project ", layout='wide')

st.title("pradnya")

sep_len=st.number_input("Sepal Length:", min_value=0.0, step=0.01)
sep_wid=st.number_input("Sepal Width:", min_value=0.0, step=0.01)
pet_len=st.number_input("Petal Length:", min_value=0.0, step=0.01)
pet_wid=st.number_input("Petal Width:", min_value=0.0, step=0.01)

# Add a button to predict
submit = st.button("Predict")

# Load the preprocessor with pickle
with open("notebook/pre.pkl", "rb") as file1:
    pre = pickle.load(file1)

# Load the model with pickle
with open('notebook/model.pkl', 'rb') as file2:
    model = pickle.load(file2)

# if submit button is pressed
if submit:
    # Convert the data into dataframe
    dct = {'sepal_length':[sep_len],
           'sepal_width':[sep_wid],
           'petal_length':[pet_len],
           'petal_width':[pet_wid]}
    # Convert above dictionary to dataframe
    xnew = pd.DataFrame(dct)
    # Transform xnew
    xnew_pre = pre.transform(xnew)
    # Predict the results with probability
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    # Print above results
    st.subheader("Predictions are : ")
    st.subheader(f"Predicted Species : {pred[0]}")
    st.subheader(f"Probability : {max_prob*100:.2f} %")
    st.progress(max_prob)









"""

application=Flask('__name__')
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_point():
    if request.method=='GET':
        return render_template('index.html')
    else:
        sepal_length=float(request.form.get('sepal_length'))
        sepal_width=float(request.form.get('sepal_width'))
        petal_length=float(request.form.get('petal_length'))
        petal_width=float(request.form.get('petal_width'))

        x_new=pd.DataFrame([sepal_length,sepal_width,petal_length,petal_width]).T
        x_new.columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        with open('pipe.pkl','rb') as file1:
            pipe=pickle.load(file1)

        x_pre=pipe.transform(x_new)
        x_pre=pd.DataFrame(x_pre,columns=x_new.columns)

        with open('./label.pkl','rb') as file2:
           le=pickle.load(file2)

        with open('./model.pkl','rb') as file3:
            model=pickle.load(file3)

        ypred=model.predict(x_pre)

        

        pred_lb=le[0]
        prob=model.predict_proba(x_pre).max()

        prediction1=f'{pred_lb} with probability: {prob:.4f}'

        return render_template('index.html',prediction=prediction1)
    
       

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)


"""
