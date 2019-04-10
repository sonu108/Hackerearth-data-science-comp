import numpy as np
from sklearn import preprocessing, cross_validation , neighbors , svm
import pandas as pd
import pylab as plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
#from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")



train_targets = df_train.P 
df_train.drop(['P'],1, inplace = True)

combined = df_train.append(df_test)

print df_train.shape,df_test.shape , combined.shape

#print combined.head()


#print combined['A'].isnull().values.ravel().sum()

columns = ('A','D','E','F','G')
for col in columns:
	dumm = pd.get_dummies(combined[col] , dummy_na=True , prefix = '{}'.format(col))
	combined.drop([col] ,1 ,inplace=True)
	combined = pd.concat([combined , dumm] , axis = 1)

M_dumm = pd.get_dummies(combined['M'] , prefix = '{}'.format(col))
combined.drop(['M'] ,1 ,inplace=True)
combined = pd.concat([combined , M_dumm] , axis = 1)

columns = ('I','J','L')
for col in columns:
	combined['{}_flag'.format(col)] = [1.0 if i=='t' else 0.0 for i in combined[col]]
	combined.drop([col] ,1, inplace=True)

combined['B'].fillna(combined['B'].median() , inplace=True)
combined['N'].fillna(combined['N'].median() , inplace=True)

combined.drop(['id'] , 1, inplace = True)



#print combined.isnull().any()

#print combined.head()

#combined = preprocessing.scale(combined)
#print combined[10]

train = combined[:552]
test = combined[552:]

#print train.shape,test.shape
accuracy = [0]*7
'''
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, train_targets)


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))
#plot.show()

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print train_reduced.shape

test_reduced = model.transform(test)
print test_reduced.shape'''

'''
for i in range(10):
	X_train, X_test , y_train ,y_test = cross_validation.train_test_split(train,train_targets,test_size=0.2)
	knn = neighbors.KNeighborsClassifier()
	logreg = LogisticRegression()
	logreg_cv = LogisticRegressionCV()
	rf = RandomForestClassifier()
	gboost = GradientBoostingClassifier()
	sv = svm.SVC()
	gnb = GaussianNB()
	models = [knn,logreg, logreg_cv, rf, gboost , sv , gnb]

	for model in models:
		clf = model.fit(X_train,y_train)
		score = clf.score(X_test, y_test)
		accuracy[models.index(model)] += score
		#print model.__class__,'accuracy = {0}'.format(score)
		#print '****'

#print [i/10 for i in accuracy]

kf = cross_validation.KFold(552 , n_folds = 10)

accuracy2 = [0]*7

for train_index,test_index in kf:
	X_train, X_test = train[train_index[0]:train_index[-1]] , train[test_index[0]:test_index[-1]]
	y_train , y_test = train_targets[train_index[0]:train_index[-1]] , train_targets[test_index[0]:test_index[-1]]

	knn = neighbors.KNeighborsClassifier()
	logreg = LogisticRegression()
	logreg_cv = LogisticRegressionCV()
	rf = RandomForestClassifier()
	gboost = GradientBoostingClassifier()
	sv = svm.SVC()
	gnb = GaussianNB()
	models = [knn,logreg, logreg_cv, rf, gboost , sv , gnb]

	for model in models:
		clf = model.fit(X_train,y_train)
		score = clf.score(X_test, y_test)
		accuracy2[models.index(model)] += score

#print [i/10 for i in accuracy2]

for i,j in enumerate(models):
	print j.__class__,accuracy[i]/10,accuracy2[i]/10
'''

'''
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validatio = cross_validation.KFold(552 , n_folds = 5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validatio,
                               verbose=1
                              )

    grid_search.fit(train, train_targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, train_targets)

'''




logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]


trained_model = []
for model in models:
	model.fit(train,train_targets)
	trained_model.append(model)

prediction = []

for model in trained_model:
	prediction.append(model.predict_proba(test)[:,1])

#print(prediction)

predictions_df = pd.DataFrame(prediction).T
predictions_df['P'] = predictions_df.mean(axis=1)


#print predictions_df

aux = pd.read_csv('test.csv')
predictions_df['id'] = aux['id']
predictions_df['P'] = predictions_df['P'].map(lambda s: 1 if s >= 0.5 else 0)

predictions_df = predictions_df[['id', 'P']]
#predictions_df.columns = ['PassengerId', 'Survived']

print predictions_df

predictions_df.to_csv('defalut_LR_LGcv_RF_GBoost.csv', index=False)


'''
output = clf.predict(test).astype(int)
df_output = pd.DataFrame()

df_output['id'] = aux['id']
df_output['P'] = output


print df_output

df_output[['id','P']].to_csv('default_LR.csv', index=False)

'''

#score = 86.23188