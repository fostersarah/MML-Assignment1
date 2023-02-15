import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
df.isna().sum()
df = df.drop(columns="Unamed")

#independent “ASR_Error,” “IntentError,” “Duration,” and “Gender,”
#dependent "Purchase"

#(1) Logistic Regression, (2) SVM, (3) Naive Bayes, and (4) Random Forest.

y = df['Purchase'].to_numpy()
X = df.drop('Purchase', axis = 1).to_numpy()

scale = StandardScaler()
scaled_X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.3)

lc = LogisticRegression()
svc = SVC(probability=True)
nbc = GaussianNB()
rfc = RandomForestClassifier()

lc.fit(X_train, y_train)
svc.fit(X_train, y_train)
nbc.fit(X_train, y_train)
rfc.fit(X_train, y_train)

y_lc_predicted = lc.predict(X_test)
y_lc_pred_proba = lc.predict_proba(X_test)

y_svc_predicted = svc.predict(X_test)
y_svc_pred_proba = svc.predict_proba(X_test)

y_nbc_predicted = nbc.predict(X_test)
y_nbc_pred_proba = nbc.predict_proba(X_test)

y_rfc_predicted = rfc.predict(X_test)
y_rfc_pred_proba = rfc.predict_proba(X_test)

print("Logistic Regression")
print(classification_report(y_test, y_lc_predicted))

print("SVC")
print(classification_report(y_test, y_svc_predicted))

print("Gaussian NB")
print(classification_report(y_test, y_nbc_predicted))

print("Random Forest Classifier")
print(classification_report(y_test, y_rfc_predicted))