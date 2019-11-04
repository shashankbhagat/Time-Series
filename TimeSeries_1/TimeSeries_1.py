import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series  		#to work on series
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA

def cls():
    print('\n'*50)
    pass


#read csv file
train=pd.read_csv('C:\MS\project\Earthquake prediction\Time Series\Train_SU63ISt.csv')
test=pd.read_csv('C:\MS\project\Earthquake prediction\Time Series\Test_0qrQsBZ.csv')

#remove Unnamed columns
train=train.drop(train.columns[train.columns.str.contains('unnamed',case=False)],axis=1)

#creating a copy to keep an original copy inplace
train_original=train.copy()
test_original=test.copy()

#check the columns available in the dataset
print(train.columns, test.columns)
print(train.dtypes, test.dtypes)

#shape of the files
print(train.shape, test.shape)

#extract datetime field and convert the datetime field in the dataset to datetime format as needed.
train['Datetime']=pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
test['Datetime']=pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M')
train_original['Datetime']=pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
test_original['Datetime']=pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M')

# as per the hypothesis for the effect of hour, day, month, year on passenger count we extract the year, month, day and hour
datasets=(train,test,train_original,test_original)
	
for i in datasets:
	i['year']=i.Datetime.dt.year
	i['month']=i.Datetime.dt.month
	i['day']=i.Datetime.dt.day
	i['hour']=i.Datetime.dt.hour

#Days 5 & 6 represent that they are weekends
train['day of week']=train['Datetime'].dt.dayofweek
temp=train['Datetime']
	
#We made a hypothesis for the traffic pattern on weekday and weekend as well. So, let’s make a weekend variable to visualize the impact of weekend on traffic.
 #first extract the day of week from Datetime and then based on the values we will assign whether the day is a weekend or not.
def applyer(row):
    if row.dayofweek==5 or row.dayofweek==6:
        return 1
    else:
        return 0
    pass

temp2=train['Datetime'].apply(applyer)
train['weekend']=temp2

#plot the time series
train.index=train['Datetime']	#indexing the Datetime to get time period on X-axis
df=train.drop('ID',1)	#drop ID to get only Datetime on X-axis

ts=df['Count']
plt.figure(figsize=(16,8))
plt.plot(ts,label='Passenger Count')
plt.title('Time Series')
plt.xlabel('Time(year-month)')
plt.ylabel('Passenger Count')
plt.legend(loc='best')
plt.show()

#Our first hypothesis was traffic will increase as the years pass by. So let’s look at yearly passenger count.
train.groupby('year')['Count'].mean().plot.bar()
plt.show()

#Our second hypothesis was about increase in traffic from May to October. So, let’s see the relation between count and month.
train.groupby('month')['Count'].mean().plot.bar()
plt.show()

#Let’s look at the monthly mean of each year separately.
temp=train.groupby(['year','month'])['Count'].mean()
temp.plot(figsize=(15,5),title='Passenger Count(Monthwise)',fontsize=14)
plt.show()

#Day wise data
train.groupby('day')['Count'].mean().plot.bar()
plt.show()

#We also made a hypothesis that the traffic will be more during peak hours. So let’s see the mean of hourly passenger count.
train.groupby('hour').mean()['Count'].plot.bar()
plt.show()

#Let’s try to validate our hypothesis in which we assumed that the traffic will be more on weekdays.
train.groupby('weekend').mean()['Count'].plot.bar()
plt.show()

#Now we will try to look at the day wise passenger count.
train.groupby('day of week').mean()['Count'].plot.bar()
plt.show()

#ID not needed in the count of passengers
train=train.drop('ID',1)

#As we have seen that there is a lot of noise in the hourly time series, we will aggregate the hourly time series to daily, weekly, and monthly time series to reduce the noise and make it more stable and hence would be easier for a model to learn.
train['Timestamp']=pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
train.index=train.Timestamp

#Hourly time series
hourly=train.resample('H').mean()
#Converting to daily time series
daily=train.resample('D').mean()
#Converting to weekly time series
weekly=train.resample('W').mean()
#Converting to monthly time series
monthly=train.resample('M').mean()

fig,axs=plt.subplots(4,1)
hourly.Count.plot(figsize=(15,8),title='Hourly',fontsize=14,ax=axs[0])
daily.Count.plot(figsize=(15,8),title='Daily',fontsize=14,ax=axs[1])
weekly.Count.plot(figsize=(15,8),title='Weekly',fontsize=14,ax=axs[2])
monthly.Count.plot(figsize=(15,8),title='Monthly',fontsize=14,ax=axs[3])
plt.show()

#We can see that the time series is becoming more and more stable when we are aggregating it on daily, weekly and monthly basis.
#But it would be difficult to convert the monthly and weekly predictions to hourly predictions, as first we have to convert the monthly predictions to weekly, weekly to daily and daily to hourly predictions, which will become very expanded process. So, we will work on the daily time series.
test['Timestamp']=pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M')
test.index=test.Timestamp

#Converting to daily mean
test=test.resample('D').mean()

train['Timestamp']=pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
train.index=train.Timestamp

#Converting to daily mean
train=train.resample('D').mean()

#The starting date of the dataset is 25-08-2012 as we have seen in the exploration part and the end date is 25-09-2014.
Train=train.ix['2012-08-25':'2014-06-24']
valid=train.ix['2014-06-25':'2014-09-25']

#check how the train and validation data has been split
Train.Count.plot(figsize=(15,8),title='Daily Ridership', fontsize=14, label='train')
valid.Count.plot(figsize=(15,8),title='Daily Ridership', fontsize=14, label='valid')
plt.xlabel('Datetime')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()

#-----------Modelling Techniques--------------
#Prediction using the naive approach
dd=np.asarray(Train.Count)
y_hat=valid.copy()
y_hat['naive']=dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(Train.index,Train['Count'],label='Train')
plt.plot(valid.index,valid['Count'],label='Valid')
plt.plot(y_hat.index,y_hat['naive'],label='Naive Forecast')
plt.legend(loc='best')
plt.title('Naive Forecast')
plt.show()


#calculate the root mean squared value
rms=sqrt(mean_squared_error(valid.Count,y_hat.naive))
print(rms)

#Prediction using the Moving Average approach
#WE would try rolling mean for last 10,20,50 days to visualise data
y_hat_avg=valid.copy()
y_hat_avg['moving_avg_forecast']=Train['Count'].rolling(10).mean().iloc[-1]
#above was the rolling avergae of lasgt 10 observations
plt.figure(figsize=(15,5))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='valid')
plt.plot(y_hat_avg['moving_avg_forecast'],label='Moving Average Forecast using 10 observations')
plt.legend(loc='best')
plt.show()

y_hat_avg=valid.copy()
y_hat_avg['moving_avg_forecast']=Train['Count'].rolling(20).mean().iloc[-1]
#above was the rolling avergae of lasgt 20 observations
plt.figure(figsize=(15,5))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='valid')
plt.plot(y_hat_avg['moving_avg_forecast'],label='Moving Average Forecast using 20 observations')
plt.legend(loc='best')
plt.show()

y_hat_avg=valid.copy()
y_hat_avg['moving_avg_forecast']=Train['Count'].rolling(50).mean().iloc[-1]
#above was the rolling avergae of lasgt 50 observations
plt.figure(figsize=(15,5))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='valid')
plt.plot(y_hat_avg['moving_avg_forecast'],label='Moving Average Forecast using 50 observations')
plt.legend(loc='best')
plt.show()

y_hat_avg=valid.copy()
y_hat_avg['moving_avg_forecast_10']=Train['Count'].rolling(10).mean().iloc[-1]
y_hat_avg['moving_avg_forecast_20']=Train['Count'].rolling(20).mean().iloc[-1]
y_hat_avg['moving_avg_forecast_50']=Train['Count'].rolling(50).mean().iloc[-1]
#above were the rolling average of last 10,20,50 observations
plt.figure(figsize=(15,5))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='valid')
plt.plot(y_hat_avg['moving_avg_forecast_10'],label='Moving Average Forecast using 10 observations')
plt.plot(y_hat_avg['moving_avg_forecast_20'],label='Moving Average Forecast using 20 observations')
plt.plot(y_hat_avg['moving_avg_forecast_50'],label='Moving Average Forecast using 50 observations')
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(valid.Count,y_hat_avg.moving_avg_forecast_50))
print(rms)

#Prediction using the Simple Exponential Smoothing approach
y_hat_avg=valid.copy()
fit2=SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES']=fit2.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='Valid')
plt.plot(y_hat_avg['SES'],label='SES')
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(valid.Count,y_hat_avg.SES))
print(rms)

#Holt's Linear Trend Model
sm.tsa.seasonal_decompose(Train.Count).plot()
result=sm.tsa.stattools.adfuller(train.Count)
plt.show()


#An increasing trend can be seen in the dataset, so now we will make a model based on the trend.
y_hat_avg=valid.copy()
fit1=Holt(np.asarray(Train['Count'])).fit(smoothing_level=0.3,smoothing_slope=0.1)
y_hat_avg['Holt_linear']=fit1.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='valid')
plt.plot(y_hat_avg['Holt_linear'],label='Holt_linear')
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(valid.Count,y_hat_avg.Holt_linear))
print(rms)

# Holt’s Linear Trend Model on daily time series
predict=fit1.forecast(len(test))
test['prediction']=predict
#above is the daily prediction.
#We would be converting this to a hourly prediction
#To do so we will first calculate the ratio of passenger count for each hour of every day
#Then we will find the average ratio of passenger count for every hour and we will get 24 ratios.
#Then to calculate the hourly predictions we will multiply the daily prediction with the hourly ratio.
#----------Let's start------------
train_original['ratio']=train_original['Count']/train_original['Count'].sum()

#Grouping the hourly ratio
temp=train_original.groupby(['hour'])['ratio'].sum()

#Groupby to csv format
pd.DataFrame(temp,columns=['hour','ratio']).to_csv('GROUPby.csv')
temp2=pd.read_csv('GROUPby.csv')
temp2=temp2.drop('hour.1',1)

#Merge Test and test_original on day, month and year
merge=pd.merge(test,test_original,on=('day','month','year'),how='left')

merge['hour']=merge['hour_y']
merge=merge.drop(['year','month','Datetime','hour_x','hour_y'],1)

#Prediction by merging merge and temp2
prediction=pd.merge(merge,temp2,on='hour',how='left')

#Converting the ratio to the original scale
prediction['Count']=prediction['prediction']*prediction['ratio']*24
prediction['ID']=prediction['ID_y']

submission=prediction.drop(['ID_x','day','ID_y','prediction','hour','ratio'],1)

#Converting the final submission to csv format
pd.DataFrame(submission,columns=['ID','Count']).to_csv('Holt_linear.csv')

#Holt's Winter model on daily time series
y_hat_avg=valid.copy()
# The idea behind Holt’s Winter is to apply exponential smoothing to the seasonal components in addition to level and trend.
# Let’s first fit the model on training dataset and validate it using the validation dataset.
fit1=ExponentialSmoothing(np.asarray(Train['Count']),seasonal_periods=7,trend='add',seasonal='add').fit()
y_hat_avg['Holt_Winter']=fit1.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(Train['Count'],label='Train')
plt.plot(valid['Count'],label='valid')
plt.plot(y_hat_avg['Holt_Winter'],label='Holt Winter')
plt.legend(loc='best')
plt.show()

rms=sqrt(mean_squared_error(valid.Count,y_hat_avg.Holt_Winter))
print(rms)


predict=fit1.forecast(len(test))
#Now we will convert these daily passenger count into hourly passenger count using the same approach which we followed above.
test['prediction']=predict
#Merge Test and test_original on day, month and year
merge=pd.merge(test,test_original,on=('day','month','year'),how='left')
merge['hour']=merge['hour_y']
merge=merge.drop(['year','month','Datetime','hour_x','hour_y'],1)

#Predicting by merging merge and temp2
prediction=pd.merge(merge,temp2,on='hour',how='left')

#Converting the ratio to the original scale
prediction['Count']=prediction['prediction']*prediction['ratio']*24

#drop unnecessary features
prediction['ID']=prediction['ID_y']
submission=prediction.drop(['day','hour','ratio','prediction','ID_x','ID_y'],1)

#Converting the final submission to csv
pd.DataFrame(submission,columns=['ID','Count']).to_csv('Holt Winter.csv')



#ARIMA - Auto Regression Integrated Moving Average
#For ARIMA, the time series must be stationary. If it is not, then we have to make it stationary
def test_stationarity(timeseries):
	#Determine rolling statistics
	rolmean=timeseries.rolling(24).mean()	# 24 hours on each day
	rolstd=timeseries.rolling(24).std()

	#Plot rolling statistics
	orig=plt.plot(timeseries,color='blue',label='original')
	mean=plt.plot(rolmean,color='red',label='Rolling Mean')
	std=plt.plot(rolstd,color='black',label='Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show(block=False)

	#Perform Dickey-Fuller test
	print('Results of Dickey-Fuller test')
	dftest=adfuller(timeseries,autolag='AIC')
	dfoutput=pd.Series(dftest[0:4],index=['Test Statistics','p-value','#Lag Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key]=value
	print(dfoutput)
    

rcParams['figure.figsize']=20,10
test_stationarity(train_original['Count'])

# Removing trends using rolling average and window size of 24
Train_log=np.log(Train['Count'])
valid_log=np.log(valid['Count'])

moving_avg=Train_log.rolling(24).mean()
plt.plot(Train_log,label='Train log')
plt.plot(moving_avg,color='red',label='moving avg')
plt.legend(loc='best')
plt.show()

# We can see the increasing trend, so let's remove this trend
train_log_moving_avg_diff=Train_log-moving_avg
train_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(train_log_moving_avg_diff)

# we also need to stabalize the mean as well.
train_log_diff=Train_log-Train_log.shift(1)
test_stationarity(train_log_diff.dropna())

# residual is the random variation in the series
# we need to decompose the series into trend and seasonality to get the residual
decomposition=seasonal_decompose(pd.DataFrame(Train_log).Count.values,freq=24)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(411)
plt.plot(Train_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#check stationarity of residual
train_log_decompose=pd.DataFrame(residual)
train_log_decompose['date']=Train_log.index
train_log_decompose.set_index('date',inplace=True)
train_log_decompose.dropna(inplace=True)
test_stationarity(train_log_decompose[0])

#Forecasting using ARIMA
#we will fit the ARIMA model and find the optimised values for p,d,q parameters.
#to find the optimised values, use the ACF(Autocorrelation Function) and PACF(Partial AutoCOrrelation Function) graph.
#ACF is the measure of correlation between time series with lasgged version of itself.
#PACF is correlation between the TimeSeries with lagged version of iteself but after eliminating the variations explained  by intervention comparisons

lag_acf=acf(train_log_diff.dropna(),nlags=25)
lag_pacf=pacf(train_log_diff.dropna(),nlags=25,method='ols')

lag_pacf=pacf(train_log_diff.dropna(),nlags=25,method='ols')

plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Autocorrelation function')
plt.show()
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray')
plt.title('Partial Autocorrelation function')
plt.show()

# p value is the lag value where the PACF chart crosses the upper confidence interval for the first time. It can be noticed that in this case p=1.
# q value is the lag value where the ACF chart crosses the upper confidence interval for the first time. It can be noticed that in this case q=1.
# Now we will make the ARIMA model as we have the p,q values. We will make the AR and MA model separately and then combine them together.

#AR model
#Auto regressive models specifies that the output variabel depends linearly on the previous output.
model = ARIMA(Train_log,order=(2,1,0)) # here the q value is zero since it is just the AR model
result_AR=model.fit(disp=-1)
plt.plot(train_log_diff.dropna(),label='original')
plt.plot(result_AR.fittedvalues,color='red',label='predictions')
plt.legend(loc='best')
plt.show()

AR_predict=result_AR.predict(start='2014-06-25',end='2014-09-25')
AR_predict=AR_predict.cumsum().shift().fillna(0)
AR_predict1=pd.Series(np.ones(valid.shape[0])*np.log(valid['Count'])[0],index=valid.index)
AR_predict1=AR_predict1.add(AR_predict,fill_value=0)
AR_predict=np.exp(AR_predict1)
plt.plot(valid['Count'],label='Valid')
plt.plot(AR_predict,color='red',label='Predict')
plt.legend(loc='best')
plt.title('RMSE: %.4f'%(np.sqrt(np.dot(AR_predict,valid['Count']))/valid.shape[0]))
plt.show()

# MA model
# the moving average model specifies tha tthe model depends linearly on the current and various past values of the stocastic term
model=ARIMA(Train_log,order=(0,1,2))	# p value is zero since it i just the MA model
result_MA=model.fit(disp=-1)
plt.plot(train_log_diff.dropna(),label='original')
plt.plot(result_MA.fittedvalues,color='red',label='prediction')
plt.legend(loc='best')
plt.show()


MA_predict=result_MA.predict(start='2014-06-25',end='2014-09-25')
MA_predict=MA_predict.cumsum().shift().fillna(0)
MA_predict1=pd.Series(np.ones(valid.shape[0])*np.log(valid['Count'])[0],index=valid.index)
MA_predict1=MA_predict1.add(MA_predict,fill_value=0)
MA_predict=np.exp(MA_predict1)
plt.plot(valid['Count'],label='valid')
plt.plot(MA_predict,color='red',label='predict')
plt.legend(loc='best')
	       
plt.title('RMSE: %.4f'%(np.sqrt(np.dot(MA_predict,valid['Count']))/valid.shape[0]))
plt.show()

# combing the above two models i.e. AR and MA
model=ARIMA(Train_log,order=(2,1,2))
result_ARIMA=model.fit(disp=-1)
plt.plot(train_log_diff.dropna(),label='original')
plt.plot(result_ARIMA.fittedvalues,color='brown',label='predicted')
plt.legend(loc='best')
plt.show()
	       
# we need to change the scale to original scale
	       
def check_prediction_diff(predict_diff,given_set):
	       predict_diff=predict_diff.cumsum().shift().fillna(0)
	       predict_base=pd.Series(np.ones(given_set.shape[0])*np.log(given_set['Count'])[0],index=given_set.index)
	       predict_log=predict_base.add(predict_diff,fill_value=0)
	       predict=np.exp(predict_log)
	       plt.plot(given_set['Count'],label='Given set')
	       plt.plot(predict,color='red',label='predict')
	       plt.legend(loc='best')
	       plt.title('RMSE:%.4f'%(np.sqrt(np.dot(predict,given_set['Count']))/given_set.shape[0]))
	       plt.show()

	       
def check_prediction_log(predict_log,given_set):
	       predict=np.exp(predict_log)
	       plt.plot(given_set['Count'],label='Predict')
	       plt.legend(loc='best')
	       plt.title('RMSE: %.4f'%(np.sqrt(np.dot(predict,given_set['Count']))/given_set.shape[0]))
	       plt.show()

	       
ARIMA_predict_diff=result_ARIMA.predict(start='2014-06-25',end='2014-09-25')
	       
check_prediction_diff(ARIMA_predict_diff,valid)
	       
check_prediction_log(ARIMA_predict_diff,valid)
	       
