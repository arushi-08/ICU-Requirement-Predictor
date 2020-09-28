from flask import Flask, stream_with_context, render_template, request, redirect, url_for, Response, jsonify
from joblib import load
import icupred
import numpy as np

# model = load("log_reg_icu_pred.joblib")
app = Flask(__name__)

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv

@app.route('/train', methods=['GET'])
def train():
    output= icupred.main()
    return Response(stream_with_context(stream_template('train.html', rows=output)))

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method =='POST':
        intubed = request.form['intubed']
        pneumonia = request.form['pneumonia']
        copd = request.form['copd']
        asthma = request.form['asthma']
        inmsupr = request.form['inmsupr']
        other_disease = request.form['other_disease']
        cardiovascular = request.form['cardiovascular']
        obesity = request.form['obesity']
        covid_res = request.form['covid_res']
        symptom_to_death = request.form['symptom_to_death']
        
        features = [[int(intubed), int(pneumonia), int(copd), int(asthma), int(inmsupr), int(other_disease), int(cardiovascular), int(obesity), int(covid_res), int(symptom_to_death)]]
        model = load('./log_reg_icu_pred.joblib')
        probabilities = model.predict_proba(features)[0]
        all_probs_new=[]
        for j in probabilities:
            all_probs_new.append(round(j,2))
        probabilities = np.array(all_probs_new)
        label_index = probabilities.argmax()
        MODEL_LABELS = ['icu','no icu']
        label = MODEL_LABELS[label_index]
        class_probabilities = dict(zip(MODEL_LABELS, probabilities.tolist()))
        return render_template('predict.html', status_code=200, text = {'status' : 'complete', 'label' : label, 'probabilities' : class_probabilities})
    return render_template('predict.html')
    
@app.route('/predict_') #, methods=['GET','POST']
def predict_():
#    if request.method == 'POST':
    intubed = request.args.get('intubed', default=2.0, type=float)
    pneumonia = request.args.get('pneumonia', default=2.0, type=float)
    copd = request.args.get('copd', default=2.0, type=float)
    asthma = request.args.get('asthma', default=2.0, type=float)
    inmsupr = request.args.get('inmsupr', default=2.0, type=float)
    other_disease = request.args.get('other_disease', default=2.0, type=float)
    cardiovascular = request.args.get('cardiovascular', default=2.0, type=float)
    obesity = request.args.get('obesity', default=2.0, type=float)
    covid_res = request.args.get('covid_res', default=2.0, type=float)
    symptom_to_death = request.args.get('symptom_to_death', default=2.0, type=float)
    features = [[intubed, pneumonia, copd, asthma, inmsupr, other_disease, cardiovascular, obesity, covid_res, symptom_to_death]]
    print(features)
    model = load('./log_reg_icu_pred.joblib')
    probabilities = model.predict_proba(features)[0]
    all_probs_new=[]
    for j in probabilities:
        all_probs_new.append(round(j,2))
    probabilities = np.array(all_probs_new)
    label_index = probabilities.argmax()
    MODEL_LABELS = ['icu','no icu']
    label = MODEL_LABELS[label_index]
    class_probabilities = dict(zip(MODEL_LABELS, probabilities.tolist()))
    return jsonify(status='complete',label=label, probabilities= class_probabilities)

@app.route('/data',methods=['GET'])
def load_data():
#    data= icupred.load_data()
#    html = data.head().to_html()
#    with open("templates/data.html", "w") as text_file:
#        text_file.write(html)
    return render_template('data.html')
    
@app.route('/', methods=['POST', 'GET'])
def home():
	return render_template('home.html')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=int("5000"),debug=True) #
