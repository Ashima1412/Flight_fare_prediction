# Flight_fare_prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Import train data
train_data = pd.read_excel('C:\\Users\\Admin\\Desktop\\ML\\Complete_project\\Data_Train.xlsx')
train_data.head()

#Import test data
test_data = pd.read_excel('C:\\Users\\Admin\\Desktop\\ML\\Complete_project\\Test_set.xlsx')
test_data.head()

train_data.info()

# Data Preprocessing
train_data['Duration'].value_counts()
train_data.dropna(inplace=True)
train_data.isnull().sum()

# Feature Engineering
train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.day
train_data['Journey_month'] = pd.to_datetime(train_data.Date_of_Journey,format='%d/%m/%Y').dt.month
train_data.head()

train_data.drop(['Date_of_Journey'],axis=1,inplace=True)

# Extracting Departure Time

train_data['Dep_hour'] = pd.to_datetime(train_data.Dep_Time).dt.hour
train_data['Dep_min'] = pd.to_datetime(train_data.Dep_Time).dt.minute

train_data.drop(['Dep_Time'],axis=1,inplace=True)

# Extracting Arrival Time

train_data['Arrival_hour'] = pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data.Arrival_Time).dt.minute
train_data.drop(['Arrival_Time'],axis=1,inplace=True)

# Assigning and converting Duration into List

duration = list(train_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' ' + '0m'
        else:
            duration[i] = '0h' +' ' + duration[i]
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))
    
    
    # Adding duration hour and min to db

train_data['Duration_hours'] = duration_hours
train_data['Duration_mins'] = duration_mins
train_data.drop(['Duration'],axis=1,inplace=True)

# Encoding Categorical Data

train_data.Airline.value_counts()

#Plotting

sns.catplot(y='Price',x='Airline',data=train_data.sort_values('Price',ascending=False),kind='boxen',height=6,aspect=3)

#Encoding Airline column - One hot Encoding(Nominal Data)

Airline = pd.get_dummies(train_data['Airline'],drop_first=True)

# Encoding Source Column

train_data.Source.value_counts()
Source = pd.get_dummies(train_data['Source'],drop_first=True)

# Encoding Destination Column

train_data.Destination.value_counts()
Destination = pd.get_dummies(train_data['Destination'],drop_first=True)

# Route and Additional info has no use in this problem

train_data.drop(['Route','Additional_Info'],axis=1,inplace=True)

# Encode Total stops - Label Encoding(Ordinal Data)
train_data.Total_Stops.value_counts()
train_data.Total_Stops.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=True)

train_data.head()

data_train = pd.concat([train_data,Airline,Source,Destination],axis=1)
data_train.head()

Test Data

# Data Preprocessing
test_data['Journey_day'] = pd.to_datetime(test_data['Date_of_Journey'],format='%d/%m/%Y').dt.day
test_data['Journey_month'] = pd.to_datetime(test_data.Date_of_Journey,format='%d/%m/%Y').dt.month
test_data.drop(['Date_of_Journey'],axis=1,inplace=True)


test_data['Dep_hour'] = pd.to_datetime(test_data.Dep_Time).dt.hour
test_data['Dep_min'] = pd.to_datetime(test_data.Dep_Time).dt.minute
test_data.drop(['Dep_Time'],axis=1,inplace=True)


test_data['Arrival_hour'] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data['Arrival_min'] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(['Arrival_Time'],axis=1,inplace=True)

duration = list(test_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' ' + '0m'
        else:
            duration[i] = '0h' +' ' + duration[i]
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))



test_data['Duration_hours'] = duration_hours
test_data['Duration_mins'] = duration_mins
test_data.drop(['Duration'],axis=1,inplace=True)

test_data.head()

# Encoding

Airline = pd.get_dummies(test_data['Airline'],drop_first=True)

Source = pd.get_dummies(test_data['Source'],drop_first=True)

Destination = pd.get_dummies(test_data['Destination'],drop_first=True)

test_data.drop(['Route','Additional_Info'],axis=1,inplace=True)

test_data.Total_Stops.value_counts()
test_data.Total_Stops.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=True)

data_test = pd.concat([test_data,Airline,Source,Destination],axis=1)

data_test.head()

data_test.drop(['Airline','Source','Destination'],axis=1,inplace=True)

# Feture Selection
X = data_train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min',
       'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins',
       'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
       'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
       'Vistara', 'Vistara Premium economy', 'Chennai', 'Delhi', 'Kolkata',
       'Mumbai', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']]
X.head()

data_train.columns

y = data_train.Price

plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(),annot=True)

# Import feature using ExtraTreeRegressor
 
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)

print(selection.feature_importances_)

# Plot the important feature

plt.figure(figsize = (8,8))
feat = pd.Series(selection.feature_importances_,index=X.columns)
feat.nlargest(20).plot(kind='barh')

Use Random Forest Regressor

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))

sns.distplot(y_test-y_pred)

from sklearn import metrics

print(metrics.mean_absolute_error(y_pred,y_test))
print(metrics.mean_squared_error(y_pred,y_test))
print(np.sqrt(metrics.mean_squared_error(y_pred,y_test)))

metrics.r2_score(y_pred,y_test)

# hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV

#No of trees in forest
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#No of features in each split
max_features = ['auto','sqrt']
# max levels in tree
max_depth = [int(x) for x in np.linspace(start=5,stop=30,num=6)]
# min no of samples required to split in each node
min_samples_split = [2,5,10,15,100]
# min no of samples required to each leaf node
min_samples_leaf = [2,5,10,15]

random_grid = {'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5)

rf_random.fit(X_train,y_train)

rf_random.best_params_

prediction = rf_random.predict(X_test)

# save model to reuse
import pickle
#open file where you want to store data
file = open('flight.pkl','wb')

#dump info to that file
pickle.dump(rf_random,file)

model = open('flight.pkl','rb')
forest = pickle.load(model)

y_prediction = forest.predict(X_test)
metrics.r2_score(y_test,y_prediction)
data_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)


