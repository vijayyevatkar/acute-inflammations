"""
Created on Wed May 15 08:08:41 2019

Dataset used: UCI Dataset Repository -> Acute Inflammations Data Set
Source: http://archive.ics.uci.edu/ml/datasets/Acute+Inflammations

Brief Description of the Dataset:

    1. Binary Classification of 2 Diseases -> acute inflammations of urinary bladder and acute nephritises
        i. Acute inflammation of urinary bladder is characterised by sudden occurrence of pains in the abdomen region and 
           the urination in form of constant urine pushing, micturition pains and sometimes lack of urine keeping. 
           Temperature of the body is rising, however most often not above 38C. The excreted urine is turbid and 
           sometimes bloody. At proper treatment, symptoms decay usually within several days. 
           However, there is inclination to returns. At persons with acute inflammation of urinary bladder, 
           we should expect that the illness will turn into protracted form. 
           
        ii. Acute nephritis of renal pelvis origin occurs considerably more often at women than at men. 
            It begins with sudden fever, which reaches, and sometimes exceeds 40C. 
            The fever is accompanied by shivers and one- or both-side lumbar pains, which are sometimes very strong. 
            Symptoms of acute inflammation of urinary bladder appear very often. 
            Quite not infrequently there are nausea and vomiting and spread pains of whole abdomen.
    
    2. Attribute Information:
        a1 Temperature of patient { 35C-42C }
        a2 Occurrence of nausea { yes, no }
        a3 Lumbar pain { yes, no }
        a4 Urine pushing (continuous need for urination) { yes, no }
        a5 Micturition pains { yes, no }
        a6 Burning of urethra, itch, swelling of urethra outlet { yes, no }
        d1 decision: Inflammation of urinary bladder { yes, no }
        d2 decision: Nephritis of renal pelvis origin { yes, no }        

@author: Vijay Yevatkar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB

def main():

    # Read the data set.
    dataset_file = "D:\\Data Science\\UCI Dataset\\Acute Inflammations Data Set\\diagnosis.data"
    header_list = ["Temp", "Nausea", "Lumbar_Pain", "Pushing", "Micturition", "Burning", "Y1", "Y2"]
    data = pd.read_csv(dataset_file, sep="\t", encoding = "utf-8", names=header_list)
    
    # As the temperature is in string format, convert it to float
    data['Temp'] = round((data['Temp'].str.replace(",",".")).astype(float),1)
    
    # As other columns are in Boolean format, convert them to 1,0
    for i in range(1,len(header_list)):
        data[header_list[i]].replace(('yes', 'no'), (1, 0), inplace=True)
    
    #data.describe()
    
    '''
    Look at the data and get an idea about the features.
    The only column which is not a Boolean is the temperature. This makes it easier for us to take decisions.
    We will use a Classification model here (Task is to decide whether it is disease or not)
    
    If you read the description, it is clear that all the features are directly correlated to the output.
    So, we will be using all the features given to us. Also, we will create 2 sub-problems from this data:
        1. Given the data -> Prdeict if the patient has Acute inflammation of urinary bladder (Y1) or not.
        2. Given the data -> Prdeict if the patient has Acute nephritis of renal pelvis (Y2) or not.
    
    Also, to begin with, we will use a Bernoulli Naive Bayes classifier as it works well when feature set is Binary.
    '''
    
    # Preprocessing - Divide the problem into 2 sub problems
    data1 = data.copy()
    y1 = data1.Y1
    X1 = data1.drop(["Y1","Y2"], axis=1)
    
    data2 = data.copy()
    y2 = data2.Y2
    X2 = data2.drop(["Y1","Y2"], axis=1)
    
    # Let's consider subproblem 1 -> Prdeict if the patient has Acute inflammation of urinary bladder (Y1) or not.
    # With X1 we will predict the against y1 which excludes y2 from the set. We use the split of 80/20
    # Note: The parameter "stratify=y1" ensures that the distribution of the +ve and -ve results in the split is optimal
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=567, stratify=y1)
    
    # Conventional method to fit a model and predict the outcome
    y1_pred_fp = fitAndPredict(X1_train, X1_test, y1_train)
    print("Subproblem 1 -> For the normal fitAndPredict method, we got the following result.")
    printResult(y1_test,y1_pred_fp)
    
    # Using the pipeline method of sklearn which takes care of the cross-validation
    # Note: You can also tune the hyperparameters with this method. 
    y1_pred_pipeline = usePipeline(X1_train, X1_test, y1_train)
    print("Subproblem 1 -> For the sklearn pipeline method, we got the following result.")
    printResult(y1_test,y1_pred_pipeline)

    # Let's consider subproblem 2 -> Prdeict if the patient has Acute nephritis of renal pelvis (Y2) or not.
    # With X2 we will predict the against y2 which excludes y1 from the set. We use the split of 80/20
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=567, stratify=y2)
    y2_pred_fp = fitAndPredict(X2_train, X2_test, y2_train)
    print("\nSubproblem 2 -> For the normal fitAndPredict method, we got the following result.")
    printResult(y2_test,y2_pred_fp)
    y2_pred_pipeline = usePipeline(X2_train, X2_test, y2_train)
    print("Subproblem 2 -> For the sklearn pipeline method, we got the following result.")
    printResult(y2_test,y2_pred_pipeline)
    
    '''
    As the feature set here is directly correlated with accurate values, we get an optimal result even with the 
    normal fitAndPredict method.
    '''

# Method to print the Accuracy results for the given test data and predicted data
def printResult(y_test, y_pred):
    print("Accuracy = {:05.2f}%. Mislabelled points = {}/{}"
          .format(100*(1-(y_test != y_pred).sum()/y_test.shape[0]),
                  (y_test != y_pred).sum(),
                  y_test.shape[0]
    ))

# Method to fit and classify the given dataset
def fitAndPredict(train_X, test_X, train_y):
    
    # StandardScaler standardized the provided data.
    # We need to make sure that the data is sclaed on the train data and the same scaler is used on the test data
    # This is to use the same means and standard deviations as we used to transform the training set. 
    # Thus, it will be a fair representation of the model
    scaler = preprocessing.StandardScaler().fit(train_X)
    X_train_scaled = scaler.transform(train_X)
    
    # Check if the scaling was correct. Mean should be centered around 0 and Standard Deviation around 1
    # print(X1_train_scaled.mean(axis=0))
    # print(X1_train_scaled.std(axis=0))
    
    X_test_scaled = scaler.transform(test_X)
    
    # Use the Bernoulli Naive Bayes from sklearn and predict the test data
    bnb = BernoulliNB()
    bnb.fit(X_train_scaled,train_y)
    return bnb.predict(X_test_scaled)

# Method to fit and classify the given dataset by using the make_pipeline method from sklearn
def usePipeline(train_X, test_X, train_y):
    
    '''
    The make_pipeline method enables us to create a pipeline with the model, which could be then reused/modified easily.
    We use this pipeline in the GridSearchCV which does the cross-validation. We use 10-fold CV below.
    We can also tune the hyperparameters using GridSearchCV, but in this case we don't need it due to the feature set.
    I have used the existing set of hyperparameters for BernoulliNB. You can obtain that using: pipeline.get_params()
    '''
    pipeline = make_pipeline(preprocessing.StandardScaler(), BernoulliNB())
    hyperParams = { 'bernoullinb__alpha': [1.0],  'bernoullinb__binarize': [0.0],
                    'bernoullinb__class_prior': [None],  'bernoullinb__fit_prior': [True]}
    clf = GridSearchCV(pipeline,hyperParams,cv=10)
    clf.fit(train_X, train_y)
    return clf.predict(test_X)

if __name__ == '__main__':
    main()
