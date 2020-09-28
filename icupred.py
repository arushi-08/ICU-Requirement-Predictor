import pandas as pd
import numpy as np
import matplotlib
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer
pd.set_option('mode.chained_assignment', None)
from joblib import dump
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import time

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.express as px

#import seaborn as sns
from matplotlib import rcParams
import plotly.graph_objects as go

import matplotlib.pyplot as plt

def load_data():
    filename="./covid.csv"
    data = pd.read_csv(filename)
    print("Loaded Patients' Dataset. Size of data: %d" % (len(data)))
    covid_data = data[data.covid_res!=3]
    print("Size of data after removing rows with awaiting covid results: %d" %(len(covid_data)))
    print(covid_data.columns)
    return covid_data
    
def preprocessing(covid_data):
    covid_data.is_copy = None
    covid_data["intubed"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["pneumonia"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["pregnancy"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["diabetes"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["copd"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["asthma"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["inmsupr"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["hypertension"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["other_disease"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["cardiovascular"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["obesity"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["renal_chronic"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["tobacco"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["contact_other_covid"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["icu"].replace({97: np.nan, 98: np.nan, 99:np.nan}, inplace=True)
    covid_data["date_died"].replace({"9999-99-99": np.nan}, inplace=True)
    
    covid_data = covid_data[covid_data.icu!=np.nan]
    covid_data["entry_date"] = pd.to_datetime(covid_data["entry_date"],dayfirst=True)
    covid_data["date_symptoms"] = pd.to_datetime(covid_data["date_symptoms"],dayfirst=True)
    covid_data["date_died"] = pd.to_datetime(covid_data["date_died"],dayfirst=True)
    covid_data['symptom_to_death'] = covid_data.date_died - covid_data.date_symptoms
    covid_data['symptom_to_death'] = covid_data['symptom_to_death'].astype('timedelta64[D]')
    
    covid_data.iloc[127706].symptom_to_death = pd.NaT
    covid_data.iloc[259415].symptom_to_death = pd.NaT
    covid_data.iloc[404958].symptom_to_death = pd.NaT
    covid_data[covid_data.sex==2.0].pregnancy = 2.0
    covid_data[covid_data.sex==2.0] = covid_data[covid_data.sex==2.0].fillna(value={'pregnancy':2.0})
    plot(covid_data)
    covid_data = covid_data.drop(['id'],axis=1)
    covid_data = covid_data.reindex(columns= ['sex','patient_type','intubed','pneumonia','pregnancy','diabetes', 'copd',
         'asthma','inmsupr','hypertension','other_disease','cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid', 'covid_res', 'age', 'symptom_to_death', 'icu'])
    covid_data = covid_data.dropna(how='any',axis=0)
    return covid_data

def select_features(icu_data):
    dataset = icu_data.values
    print(icu_data.columns)
    X = dataset[:,:-1]
    y = dataset[:,-1]
    fs = SelectKBest(score_func=mutual_info_classif, k=10)
    X = fs.fit_transform(X, y)
    return icu_data, X, y, fs

def plot(covid_data):
    deaths_by_date = covid_data.groupby('date_died')['id'].agg('count').to_frame(name='count')
    plt.figure(figsize=(12,4))
    plt.title("Patients' Death Plot")
    plt.scatter(x=deaths_by_date.index,y=deaths_by_date['count'],edgecolor='k',color='lightgreen',s=100)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("static/images/deathplot.png", bbox_inches = 'tight')

    monthly_df = covid_data.groupby([covid_data['entry_date'].dt.to_period("M").astype(str),'icu','covid_res'])['id'].agg('count').to_frame(name='count').reset_index()
    monthly_df = monthly_df[1:]
    y = monthly_df[(monthly_df['icu']==1) & (monthly_df['covid_res']==1)]['count']
    y2 = monthly_df[(monthly_df['icu']==1) & (monthly_df['covid_res']==2)]['count']
    x = monthly_df[(monthly_df['icu']==1)  & (monthly_df['covid_res']==1)]['entry_date']
    if x.nunique!=1 and y.nunique!=1:
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(111)
        plt.title("ICU Patients")
        res1 = ax.bar(x,height=y,color='red')
        res2 = ax.bar(x,height=y2,color='lightgreen')
        ax.grid(True)
        ax.set_ylabel('Patients')
        ax.set_xticklabels(('Feb','Mar','May','Jun'))
        ax.legend( (res1[0], res2[0]), ('covid_res +', 'covid_res -') )
        plt.savefig("static/images/covid_icu.png", bbox_inches = 'tight')


@ignore_warnings(category=ConvergenceWarning)
def modelrun(icu_data, X, y):
    model = LogisticRegression()
    accuracy = cross_val_score(model, X, y, cv=5)
    precision = cross_val_score(model, X, y, cv=5, scoring='precision_micro')
    recall = cross_val_score(model, X, y, cv=5, scoring='recall_micro')
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1_micro')
    print("Model Scores: %0.2f Accuracy | %0.2f Precision | %0.2f Recall | %0.2f F1" %(accuracy.mean(), precision.mean(), recall.mean(), f1.mean()))

    return model, round(accuracy.mean(),2), round(precision.mean(),2), round(recall.mean(),2), round(f1.mean(),2)

def prep_test_cases(all_features, all_probs, feature_names, target_names):
    all_test_cases = []
    for feat_vec, prob_vec in zip(all_features, all_probs):
        feat_dict = {name:val for name, val in zip(feature_names, feat_vec) if not val is None}
        prob_dict = dict(zip(target_names, prob_vec))
        expected_label = target_names[prob_vec.argmax()]
        expected_response = dict(label=expected_label,
                                 probabilities=prob_dict,
                                 status='complete')
        test_case = dict(features=feat_dict,
                         expected_status_code=200,
                         expected_response=expected_response)
        all_test_cases.append(test_case)
    return all_test_cases

def main():
    yield( "Step 1: Loading Data")
    covid_data = load_data()
    yield ("Loaded Patients' Dataset. Size of data: %d" % (len(covid_data)))
    yield ("Step 2: Cleaning Data")
    icu_data = preprocessing(covid_data)
    yield ("Step 3: Selecting top 10 features")
    icu_data, X, y, fs = select_features(icu_data)

    time.sleep(2)
    
    plt.figure(figsize=(15,5))
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_, align='edge',width=0.3)
    features = ['sex', 'patient_type', 'intubed', 'pneumonia', 'pregnancy', 'diabetes',
       'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease',
       'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
       'contact_other_covid', 'covid_res', 'age', 'symptom_to_death']
    pos = np.arange(len(features))
    plt.tight_layout()
    plt.xlabel("Data Features")
    plt.ylabel("SelectKBest Scores")
    plt.xticks(pos, features, rotation=45)
    plt.savefig("static/images/plt.png", bbox_inches = 'tight')
    yield ("Plot ready")
    time.sleep(2)
    yield ("Step 4: Resampling Data due to Imbalanced target classes: %d patients who require ICU, %d patients who don't." %(len(icu_data[icu_data.icu==1.0]), len(icu_data[icu_data.icu==2.0])))
    time.sleep(2)
    #define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy=0.9)
    X, y = oversample.fit_resample(X, y)
    #define oversampling strategy
    undersample = RandomUnderSampler(sampling_strategy=0.9)
    X, y = undersample.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    yield ("Step 5: Running Logistic Regression Model")
    model, accuracy, precision, recall, f1 = modelrun(icu_data, X, y)
    yield ("Score on the Testing Set: Accuracy ",accuracy, "Precision ", precision, "Recall ", recall, "F1 ", f1)
    yield ("Saving model to log_reg_icu_pred.joblib")
    # print("----x----x----x----x-----")
    print(X_train.shape)
    model.fit(X_train,y_train)
    dump(model, filename="./log_reg_icu_pred.joblib")
    yield ("Check 1")
    all_probs = model.predict_proba(X_test)
    print(all_probs, type(all_probs))
    all_probs_new=[]
    for j in all_probs:
        prob_vec=[]
        for i in j:
            prob_vec.append(round(i,2))
        all_probs_new.append(prob_vec)
    all_probs_new = np.array(all_probs_new)
    yield ("Check 2")
    all_test_cases = prep_test_cases(X_test,
                                     all_probs_new,
                                     ['intubed','pneumonia','copd','asthma','inmsupr', 'other_disease', 'cardiovascular', 'obesity', 'covid_res', 'symptom_to_death'],
                                     ['icu','no icu'])
    yield ("Check 3")
    with open('testdata.json', 'w') as fout:
        json.dump(all_test_cases, fout)

#if __name__ == "__main__":
#    main()
