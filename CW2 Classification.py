#%% Import Library
import pandas as pd
from pandas import set_option
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot
#%% Reading CSV
datas = pd.read_csv('Credit-Scoring.csv')
datas.head(20)
#%% Removing Feature that not Relevant
columns_to_remove = ['ID','Customer_ID','Month', 'Name', 'SSN', 'Payment_of_Min_Amount', 'Num_Credit_Card', 'Num_Credit_Inquiries', 'Delay_from_due_date', 'Changed_Credit_Limit']
data = datas.drop(columns=columns_to_remove)
data.head(10)
#%% Dimensions of data
shape = data.shape
print(shape)
#%% Attributes Data Types
types = data.dtypes
print(types)
#%% Peeking Null Value
nullv = data.isnull().sum()
nullv
# %% Cleaning the Data
data['Annual_Income'] = data['Annual_Income'].astype(str).str.replace('_', '')
data['Annual_Income'] = pd.to_numeric(data['Annual_Income'], errors='coerce')
data['Outstanding_Debt'] = data['Outstanding_Debt'].astype(str).str.replace('_', '')
data['Outstanding_Debt'] = pd.to_numeric(data['Outstanding_Debt'], errors='coerce')
data['Num_of_Delayed_Payment'] = data['Num_of_Delayed_Payment'].astype(str).str.replace('_', '')
data['Num_of_Delayed_Payment'] = pd.to_numeric(data['Num_of_Delayed_Payment'], errors='coerce')
data['Num_of_Loan'] = data['Num_of_Loan'].astype(str).str.replace('_', '')
data['Num_of_Loan'] = pd.to_numeric(data['Num_of_Loan'], errors='coerce')
data['Amount_invested_monthly'] = data['Amount_invested_monthly'].astype(str).str.replace('_', '')
data['Amount_invested_monthly'] = pd.to_numeric(data['Amount_invested_monthly'], errors='coerce')
data['Monthly_Balance'] = data['Monthly_Balance'].astype(str).str.replace('_', '')
data['Monthly_Balance'] = pd.to_numeric(data['Monthly_Balance'], errors='coerce')
data['Age'] = data['Age'].astype(str).str.replace('_', '')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data[data['Age'] >= 0]
mask = (data['Age'] > 0) & (data['Age'] < 100)
data = data[mask]
set_option('display.width', 100)
set_option('display.precision', 3)
description = data.describe()
print(description)
#%% Printing Unique Value from String
unique_values = data['Credit_Score'].unique()
print(unique_values)

unique_values = data['Occupation'].unique()
print(unique_values)

unique_values = data['Credit_Mix'].unique()
print(unique_values)

unique_values = data['Type_of_Loan'].unique()
print(unique_values)

unique_values = data['Payment_Behaviour'].unique()
print(unique_values)
#%% Cleaning the Data by removing specific value
data = data.dropna()
data.drop_duplicates()
data = data[data['Occupation'] != '_______']
data = data[data['Credit_Mix'] != '_']
data = data[data['Type_of_Loan'] != 'Not Specified']
data = data[data['Payment_Behaviour'] != '!@9#%8']
label_encoder = LabelEncoder()
#%% Converting String Value into Integer
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['Credit_Mix'] = label_encoder.fit_transform(data['Credit_Mix'])
data['Type_of_Loan'] = label_encoder.fit_transform(data['Type_of_Loan'])
data['Credit_History_Age'] = label_encoder.fit_transform(data['Credit_History_Age'])
data['Payment_Behaviour'] = label_encoder.fit_transform(data['Payment_Behaviour'])
data.replace(to_replace ="Good", value = 0, inplace = True)
data.replace(to_replace ="Standard", value = 1, inplace = True)
data.replace(to_replace ="Poor", value = 2, inplace = True)
data.head(10)
# %% Splitting Data
selected_features = ['Age', 'Monthly_Inhand_Salary', 'Annual_Income', 'Credit_Utilization_Ratio', 'Outstanding_Debt', 'Occupation', 'Credit_Mix', 'Type_of_Loan', 'Num_Bank_Accounts', 'Total_EMI_per_month', 'Interest_Rate','Num_of_Delayed_Payment', 'Credit_History_Age']
X_train = data[selected_features]
y_train = data['Credit_Score']
X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size=0.3, random_state=7)
#%% Modelling
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('RF',RandomForestClassifier(n_estimators=100)))
#%% Printing the Result from Training
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 7, shuffle=True)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" %(name,cv_results.mean(),cv_results.std())
    print(msg)
#%% Compare algorithms
fig = pyplot.figure(figsize = (10,10))
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
#%% Validating with Test Data
rf = RandomForestClassifier(random_state=7)
rf.fit(X_train, y_train)
predictions = rf.predict(X_val)
print(accuracy_score(y_val, predictions))
print(confusion_matrix(y_val, predictions))
print(classification_report(y_val, predictions))