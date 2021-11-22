'''
COMP721: Machine Learning
Project: NBA Prediction
Author:  217008217
University of KwaZulu-Natal
School of Mathematics, Statistics and Computer Science
'''

import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm


data = pd.read_csv('team_season.csv')
data = data[data.year > 1980] #remove data that occured prior to 1980


#Calculate and display Pearson correlation coefficients to aid in feature selection 
cor = data.corr()
sns.heatmap(cor, xticklabels=True, yticklabels=True)
plt.show()


#Split data into Training and Testing sets, Data with year 2004 is used for testing
X_train = data.iloc[:len(data)-30, [3, 4, 5, 7, 10, 13, 15, 18, 19, 20, 28, 29, 30, 34, 35]]
X_test = data.iloc[len(data)-30:, [3, 4, 5, 7, 10, 13, 15, 18, 19, 20, 28, 29, 30, 34, 35]]


#Create a new feature called Win Ratio, where Win Ratio = Won + (Won + Lost)
X_train_win_ratio = X_train.iloc[:, [13, 14]]
X_train_win_ratio = X_train_win_ratio.values
X_test_win_ratio = X_test.iloc[:, [13, 14]]
X_test_win_ratio = X_test_win_ratio.values


#Add the Win Ratios as the y_test and y_train values
target = [] 
for i in range(0, len(X_train_win_ratio)):
    target.append(X_train_win_ratio[i][0] / (X_train_win_ratio[i][0] + X_train_win_ratio[i][1]))
X_train = X_train.drop(['won', 'lost'], 1)
y_train = pd.DataFrame(target)
y_train.columns = ['Win Ratio']

target = []
for i in range(0, len(X_test_win_ratio)):
    target.append(X_test_win_ratio[i][0] / (X_test_win_ratio[i][0] + X_test_win_ratio[i][1]))
X_test = X_test.drop(['won', 'lost'], 1)
y_test = pd.DataFrame(target)
y_test.columns = ['Win Ratio']



######## Model 1: Linear Regression ######## 

from sklearn.linear_model import LinearRegression
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
X_test2 = X_test.copy(deep=True)
X_test2['LrPrediction'] = y_pred

print("\nLinear Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


######## Model 2: LASSO Regression ######## 

from sklearn.linear_model import Lasso
Lasso_model = Lasso()
Lasso_model.fit(X_train, y_train)
y_pred = Lasso_model.predict(X_test)
X_test2['LassoPrediction'] = y_pred

print("\nLASSO Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


######## Model 3: k-Nearest Neighbors Regression ######## 

from sklearn.neighbors import KNeighborsRegressor
kNN_model = KNeighborsRegressor(n_neighbors=6)
kNN_model.fit(X_train, y_train.values.ravel())
y_pred = kNN_model.predict(X_test)
X_test2['KnnPrediction'] = y_pred

print("\nk-Nearest Neighbors Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


######## Model 4: Support Vector Regression ######## 

from sklearn.svm import SVR
SVR_model = SVR(kernel='poly')
SVR_model.fit(X_train, y_train.values.ravel())
y_pred = SVR_model.predict(X_test)
X_test2['SvrPrediction'] = y_pred

print("\nSupport Vector Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


######## Model 5: Random Forest Regression ######## 

from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor(n_estimators=50, random_state=0)
RF_model.fit(X_train, y_train.values.ravel())
y_pred = RF_model.predict(X_test)
X_test2['RfPrediction'] = y_pred

print("\nRandom Forest Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


######## Model 6: Bayesian Ridge Regression ######## 

from sklearn.linear_model import BayesianRidge
BRR_model = BayesianRidge()
BRR_model.fit(X_train, y_train.values.ravel())
y_pred = BRR_model.predict(X_test)
X_test2['BbrPrediction'] = y_pred

print("\nBayesian Ridge Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


######## Model 7: Least Angle Regression (LARS) ######## 

from sklearn.linear_model import LassoLars
Lars_model = LassoLars(alpha=.1, normalize=False)
Lars_model.fit(X_train, y_train)
y_pred = Lars_model.predict(X_test)
X_test2['LarsPrediction'] = y_pred

print("\nLARS Regression Performance:")
print("Mean Absolute Error: ", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean Squared Error: ", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Max Error: ", round(sm.max_error(y_test, y_pred), 2))
print("R-Squared Score: ", round(sm.r2_score(y_test, y_pred), 2))


#Predict Game Outcomes
X_test2['team'] = data['team']
X_test2['Label'] = y_test.values

print("Game Outcomes:")
team_names = ['ATL','BOS','CHI','CHR','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NJN','NOH','NYK','ORL','PHI','PHO','POR','SAC','SAS','SEA','TOR','UTA','WAS']
model_names = ['LrPrediction', 'LassoPrediction', 'KnnPrediction', 'SvrPrediction', 'RfPrediction', 'LarsPrediction', 'BbrPrediction']

#predict the outcome between every team-pair in the test data
for i in range(0, len(team_names) - 1):
    for j in range(i + 1, len(team_names)):
        team1name = team_names[i]
        team2name = team_names[j] 
        print('Match between: ', team1name, ' and ', team2name)
        team1 = X_test2[X_test2['team']==team1name]
        team2 = X_test2[X_test2['team']==team2name]
        for k in model_names:
            print('Model ', k, ' Predicts: ')
            team1WinRatio = team1[k].values
            team2WinRatio = team2[k].values            
            if team1WinRatio > team2WinRatio:
                print(team1name, "beats", team2name)
            elif team1WinRatio < team2WinRatio:
                print(team2name, "beats", team1name)
            else:
                print('Draw!')
        print('\n')


#Plot the models and actual values 
game = [i for i in range(0, 30)]
plt.plot(game, y_test, label='Actual Values', color='red')
plt.xlabel('Team')
plt.ylabel('Win Ratio Score')
plt.plot(game, X_test2['LrPrediction'], label='Linear Regression', color='blue')
plt.plot(game, X_test2['LassoPrediction'], label='LASSO', color='green')
plt.plot(game, X_test2['KnnPrediction'], label='k-Nearest Neighbors', color='black')
plt.plot(game, X_test2['SvrPrediction'], label='Support Vector Regression', color='orange')
plt.plot(game, X_test2['RfPrediction'], label='Random Forest', color='purple')
plt.plot(game, X_test2['LarsPrediction'], label='LARS Regression', color='yellow')
plt.plot(game, X_test2['BbrPrediction'], label='Bayesian Ridge', color='pink')

#plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
plt.show()
