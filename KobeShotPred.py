# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:36:43 2016

@author: meng
"""

#--------------- import library --------------------------

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib
import operator
from matplotlib import pylab as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

#------------------------- load data -----------------------------------#
data = pd.read_csv('C:/HKUST-WM/Research Related/Machine Learning/kaggle/Kobe Bryont Shot Selection/data.csv') # the train dataset is now a Pandas DataFrame
#test = pd.read_csv('C:/HKUST-WM/Research Related/Machine Learning/kaggle/Kobe Bryont Shot Selection/test.csv') # the train dataset is now a Pandas DataFrame
#train = data.dropna(subset = 'shot_made_flag')

train = data[data.shot_made_flag.notnull()]
test = data[data.shot_made_flag.isnull()]


#------------------ Some simple study of pct vs shot-dis ----------------
shot_distance_train = train.shot_distance
unique_shot_dis_train = shot_distance_train.unique()
shot_pct = []

for dis in unique_shot_dis_train:
    shot_dis_pct = train.shot_made_flag[train.shot_distance == dis].mean()
    shot_pct.append(shot_dis_pct)

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(unique_shot_dis_train,shot_pct,'b*')
shot_distance_train.hist(bins=100)

#--------------------  mins/seonnds remaining -------------------------------------------------
mins_remaining = train.minutes_remaining
mins_remaining.hist() 
"""quantitative variable can be directly plotted using hist() command
    but for quanlitative variable, should use value_counts(), then plot """
shot_mins_pct = []

uniq_mins_remaining = mins_remaining.unique()
uniq_mins_remaining.sort()
for mins in uniq_mins_remaining:
    shotmins_pct = train.shot_made_flag[train.minutes_remaining == mins].mean()
    shot_mins_pct.append(shotmins_pct)

shot_mins_pct=pd.Series(shot_mins_pct,index=uniq_mins_remaining)
shot_mins_pct.plot(kind = 'bar')


"""Seconds remaining"""
secs_remaining = train.seconds_remaining
secs_remaining.hist(bins= 60)
"As seconds remain smaller, attempts increases"
shot_secs_pct = []

uniq_secs_remaining = secs_remaining.unique()
uniq_secs_remaining.sort()
for secs in uniq_secs_remaining:
    shotsecs_pct = train.shot_made_flag[train.seconds_remaining == secs].mean()
    shot_secs_pct.append(shotsecs_pct)
    
shot_secs_pct = pd.Series(shot_secs_pct,index = uniq_secs_remaining)
shot_secs_pct.plot(kind='bar')
"As seconds remain smaller, shot pct decreases"

"""Create new variable: seconds remaining for this period """
train['secondsFromPeriodEnd'] = 60*train['minutes_remaining']+train['seconds_remaining']
xx = 60*(11-train['minutes_remaining'])+(60-train['seconds_remaining'])
train['secondsFromPeriodStart'] = xx
xx = (train['period'] <= 4).astype(int)*(train['period']-1)*12*60 + (train['period'] > 4).astype(int)*((train['period']-4)*5*60 + 3*12*60) + train['secondsFromPeriodStart']
train['secondsFromGameStart'] = xx

"plot the accuracy as a function of time"

plt.rcParams['figure.figsize'] = (15, 10)

binSizeInSeconds = 20
timeBins = np.arange(0,60*(4*12+3*5),binSizeInSeconds)+0.01
attemptsAsFunctionOfTime, b = np.histogram(train['secondsFromGameStart'], bins=timeBins)     
madeAttemptsAsFunctionOfTime, b = np.histogram(train.ix[train['shot_made_flag']==1,'secondsFromGameStart'], bins=timeBins)     
accuracyAsFunctionOfTime = madeAttemptsAsFunctionOfTime.astype(float)/attemptsAsFunctionOfTime
#accuracyAsFunctionOfTime[attemptsAsFunctionOfTime <= 50] = 0 # zero accuracy in bins that don't have enough samples

maxHeight = max(attemptsAsFunctionOfTime) + 30
barWidth = 0.999*(timeBins[1]-timeBins[0])
 
plt.figure()
plt.subplot(2,1,1)
plt.bar(timeBins[:-1],attemptsAsFunctionOfTime, align='edge', width=barWidth)
plt.xlim((-20,3200))
plt.ylim((0,maxHeight))
plt.ylabel('attempts')
plt.title(str(binSizeInSeconds) + ' second time bins')
plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0,ymax=maxHeight, colors='r')
plt.subplot(2,1,2)
plt.bar(timeBins[:-1],accuracyAsFunctionOfTime, align='edge', width=barWidth) 
plt.xlim((-20,3200))
plt.ylabel('accuracy')
plt.xlabel('time [seconds from start of game]')
plt.vlines(x=[0,12*60,2*12*60,3*12*60,4*12*60,4*12*60+5*60,4*12*60+2*5*60,4*12*60+3*5*60], ymin=0.0,ymax=0.7, colors='r')


#---------------------- shot type pct -----------------------------
shot_type = train.combined_shot_type
uniq_shot_type = shot_type.unique()
shot_type_pct = []

for shottype in uniq_shot_type:
    shottype_pct = train.shot_made_flag[train.combined_shot_type == shottype].mean()
    shot_type_pct.append(shottype_pct)

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

#ax1.plot(uniq_shot_type,shot_type_pct,'b*')
#shot_type.hist()

""""The above plot command won't work, because shot_type is categorical variable
     So change to value_counts"""

shot_type_counts = shot_type.value_counts()


#--------------------- Opponent pct ---------------------------------------#
opponent = train.opponent
shot_count_oppo = opponent.value_counts()
opponent.value_counts().plot(kind='bar')
uniq_oppo = opponent.unique()

shot_oppo_pct = []
for oppo in uniq_oppo:
    shotpct = train.shot_made_flag[train.opponent==oppo].mean()
    shot_oppo_pct.append(shotpct)
    
shot_oppo_pct = pd.Series(shot_oppo_pct,index = uniq_oppo)
shot_oppo_pct.plot(kind='bar')

""" Didn't see much difference for shot pct for different opponent 
    min is 0.400, max = 0.477 """
worst_oppo = np.argmin(shot_oppo_pct) # find the index with smallest val
best_oppo = np.argmax(shot_oppo_pct) # find the index with largest val


# ------------------------- Season effect --------------------------------#
season = train.season
shot_count_season = season.value_counts(sort=False)
shot_count_season = shot_count_season.sort_index()
shot_count_season.plot(kind='bar')

shot_season_pct = []
uniq_season = season.unique()
uniq_season.sort()
for sn in uniq_season:
    shotpct = train.shot_made_flag[train.season == sn].mean()
    shot_season_pct.append(shotpct)
    
shot_season_pct = pd.Series(shot_season_pct,index = uniq_season)
shot_season_pct.plot(kind='bar')

""" Shot pct decrease in the recent years """

# ------------------------- PlayOffs ----------------------------------#
regular = train[train.playoffs==0]
playoffs = train[train.playoffs==1]

regular_pct = regular.shot_made_flag.mean()
playoff_pct = playoffs.shot_made_flag.mean()
""" shot pct not much diff between regular and playoff """

# --------------------- shot_zone_range -------------------------------#
"""shot_zone_range should be very similar to shot_distance, but now with 
 smaller number of categories """

shot_range_train = train.shot_zone_range
unique_shot_range_train = shot_range_train.unique()
shot_range_pct = []

for rge in unique_shot_range_train:
    shot_rge_pct = train.shot_made_flag[train.shot_zone_range == rge].mean()
    shot_range_pct.append(shot_rge_pct)

fig = plt.figure()
shot_range_train.value_counts().plot(kind = 'bar')

shot_range_pct = pd.Series(shot_range_pct,index = unique_shot_range_train)
shot_range_pct.plot(kind='bar')
 

#------------------ Period effect ---------------------------------------#
period_train = train.period
period_train.value_counts().plot(kind = 'bar')

uniq_period = period_train.unique()
shot_period_pct = []

for prd in uniq_period:
    shot_prd_pct = train.shot_made_flag[train.period == prd].mean()
    shot_period_pct.append(shot_prd_pct)
    
shot_period_pct = pd.Series(shot_period_pct,index = uniq_period)
shot_period_pct.plot(kind='bar')








###################### Model Building #######################################
#-------------------- Simple Random Forest -------------------------------#





#-------------------- Simple XGBoost -------------------------------------------#
num_round = 50						  
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.03
params["subsample"] = 0.8
params["colsample_bytree"] = 0.7
params["silent"] = 1
params["max_depth"] = 5
params["min_child_weight"] = 1
params["eval_metric"] = "logloss"

testID = test['shot_id']

"Feature Importance"
train_Y = train.shot_made_flag
train_X = train
train_X.drop('shot_made_flag', inplace=True, axis=1)
#train_X.drop('matchup', inplace=True, axis=1) " contains home or guest info"
train_X.drop('team_name', inplace=True, axis=1)
train_X.drop('team_id', inplace=True, axis=1)
train_X.drop('game_event_id', inplace=True, axis=1)
train_X.drop('game_id', inplace=True, axis=1)
train_X.drop('shot_id', inplace=True, axis=1)

test.drop('shot_made_flag', inplace=True, axis=1)
test.drop('team_name', inplace=True, axis=1)
test.drop('team_id', inplace=True, axis=1)
test.drop('game_event_id', inplace=True, axis=1)
test.drop('game_id', inplace=True, axis=1)
test.drop('shot_id', inplace=True, axis=1)


#train_X['secondsFromPeriodEnd'] = 60*train_X['minutes_remaining']+train_X['seconds_remaining']
#xx = 60*(11-train_X['minutes_remaining'])+(60-train_X['seconds_remaining'])
#train_X['secondsFromPeriodStart'] = xx
#xx = (train_X['period'] <= 4).astype(int)*(train_X['period']-1)*12*60 + (train_X['period'] > 4).astype(int)*((train_X['period']-4)*5*60 + 3*12*60) + train_X['secondsFromPeriodStart']
#train_X['secondsFromGameStart'] = xx
#
#test['secondsFromPeriodEnd'] = 60*test['minutes_remaining']+test['seconds_remaining']
#xx = 60*(11-test['minutes_remaining'])+(60-test['seconds_remaining'])
#test['secondsFromPeriodStart'] = xx
#xx = (test['period'] <= 4).astype(int)*(test['period']-1)*12*60 + (test['period'] > 4).astype(int)*((test['period']-4)*5*60 + 3*12*60) + test['secondsFromPeriodStart']
#test['secondsFromGameStart'] = xx


"Categorical Variable has to be changed to numerical in order to use xgboost"
featureNames = train_X.columns[0:len(train_X.columns)]
for c in featureNames:
    if train_X[c].dtype.name =='object':
        le = LabelEncoder()
        le.fit(np.append(train_X[c],test[c]))
        train_X[c] = le.transform(train_X[c]).astype(int)
        test[c] = le.transform(test[c]).astype(int)

dtrain = xgb.DMatrix(train_X,label=train_Y)
dtest = xgb.DMatrix(test)

gbdt = xgb.train(params, dtrain, num_round)

"Feature importance"
importance = gbdt.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

"prediction"
test_preds = gbdt.predict(dtest)

submission = pd.DataFrame({"shot_id": testID, "shot_made_flag": test_preds})
submission.to_csv("FirstSubmission.csv", index=False)




# --------------------------- SVM ---------------------------------------#






