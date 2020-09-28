import pandas as pd
import numpy as np
import matplotlib
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

def load_data():
    filename="./covid.csv"
    data = pd.read_csv(filename)
    print("Loaded Patients' Dataset. Size of data: %d" % (len(data)))
    covid_data = data[data.covid_res!=3]
    print("Size of data after removing rows with awaiting covid results: %d" %(len(covid_data)))
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
    covid_data = covid_data.drop(['id'],axis=1)
    covid_data = covid_data[covid_data.icu!=np.nan]
    covid_data["entry_date"] = pd.to_datetime(covid_data["entry_date"],dayfirst=True)
    covid_data["date_symptoms"] = pd.to_datetime(covid_data["date_symptoms"],dayfirst=True)
    covid_data["date_died"] = pd.to_datetime(covid_data["date_died"],dayfirst=True)
    covid_data['symptom_to_death'] = covid_data.date_died - covid_data.date_symptoms
    covid_data['symptom_to_death'] = covid_data['symptom_to_death'].astype('timedelta64[D]')
    covid_data.iloc[127706].symptom_to_death = pd.NaT
    covid_data.iloc[259415].symptom_to_death = pd.NaT
    covid_data.iloc[404958].symptom_to_death = pd.NaT
    covid_data = covid_data.drop(['entry_date', 'date_symptoms', 'date_died'],axis=1)
    covid_data[covid_data.sex==2.0].pregnancy = 2.0
    covid_data[covid_data.sex==2.0] = covid_data[covid_data.sex==2.0].fillna(value={'pregnancy':2.0})
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


@ignore_warnings(category=ConvergenceWarning)
def modelrun(icu_data, X, y):
    dataset = icu_data.values
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    #define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy=0.9)
    X, y = oversample.fit_resample(X, y)
    #define oversampling strategy
    undersample = RandomUnderSampler(sampling_strategy=0.9)
    X, y = undersample.fit_resample(X, y)
    model = LogisticRegression()
    
#     model = MLPClassifier(random_state=1, max_iter=300)
    accuracy = cross_val_score(model, X, y, cv=5)
    precision = cross_val_score(model, X, y, cv=5, scoring='precision_micro')
    recall = cross_val_score(model, X, y, cv=5, scoring='recall_micro')
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1_micro')
    print("Model Scores: %0.2f Accuracy | %0.2f Precision | %0.2f Recall | %0.2f F1" %(accuracy.mean(), precision.mean(), recall.mean(), f1.mean()))
    # print('Confusion matrix:')
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    cv_results = cross_validate(model.fit(X, y), X, y, cv=5, scoring=scoring)
    model.fit(X_train,y_train)
    # dump(model, filename="./log_reg_icu_pred.joblib")
    # print(cv_results)
    return round(accuracy.mean(),2), round(precision.mean(),2), round(recall.mean(),2), round(f1.mean(),2)

def main():
    yield "Step 1: Loading Data"
    covid_data = load_data()
    yield ("Loaded Patients' Dataset. Size of data: %d" % (len(covid_data)))
    yield "Step 2: Cleaning Data"
    icu_data = preprocessing(covid_data)
    yield "Step 3: Selecting top 10 features"
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
    yield "Plot ready"
    time.sleep(2)
    yield ("Step 4: Resampling Data due to Imbalanced target classes: %d patients who require ICU, %d patients who don't." %(len(icu_data[icu_data.icu==1.0]), len(icu_data[icu_data.icu==2.0])))
    time.sleep(2)
    yield "Step 5: Running Logistic Regression Model"
    accuracy, precision, recall, f1 = modelrun(icu_data, X, y)
    yield ("Accuracy ",accuracy, "Precision ", precision, "Recall ", recall, "F1 ", f1)
    # print("----x----x----x----x-----")

#if __name__ == "__main__":
 #   main()

