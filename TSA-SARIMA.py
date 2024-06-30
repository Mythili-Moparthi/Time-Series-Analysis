#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


# In[99]:


df = pd.read_csv('/Users/mythilimoparthi/Desktop/Case Study Data.csv')


# In[100]:


df.info()


# In[101]:


df.shape


# In[102]:


df.columns


# In[103]:


# finding max and min values in the emp column
max = df['emp'].max()
min = df['emp'].min()
print("max value :",max,
      "min value :",min)


# In[104]:


#outlier detection through IQR

Q1 = df['emp'].quantile(0.25)
Q3 = df['emp'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = (Q1 - 1.5 * IQR).round(2)
upper_bound = (Q3 + 1.5 * IQR).round(2)
print("Lower Bound:",lower_bound, "Upper Bound:",upper_bound)

outliers = df[(df['emp'] < lower_bound) | (df['emp'] > upper_bound)] #outlier identification

print("Outliers:",outliers)


# In[105]:


df.isna().sum()


# In[106]:


df['month'] = pd.to_datetime(df['month'])
df['Year'] = df['month'].dt.year


# In[107]:


df['Year'].nunique()


# In[108]:


# To check if there is missing value in the month column
yearly_counts = df.groupby(df['Year']).size()
print(yearly_counts)


# In[10]:


df.head()


# In[11]:


df.info()


# # EDA

# In[109]:


#visualizing the data through years
df.plot(x='month', y='emp', figsize=(10,5), color='darkorange',grid=True)
plt.grid(alpha=0.5)
plt.xlabel('Date')
plt.ylabel('emp')
plt.title('emp data through years')
plt.show()


# In[110]:


# yearwise analysis of data
yearly_emp = df.groupby('Year')['emp'].sum()

plt.figure(figsize=(17, 6))
ax = yearly_emp.plot(kind='bar', color='coral')
plt.title('Yearly Employment')
plt.xlabel('Year')
plt.ylabel('Total Employment')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

#to add value labels on top
for bar in ax.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f'{bar.get_height():.0f}', ha='center', va='bottom', color='black',rotation = 45)

plt.tight_layout()
plt.show()


# In[111]:


yearly_totals = df.groupby('Year')['emp'].sum().reset_index()
yearly_mean = df.groupby('Year')['emp'].mean().reset_index()

# visualizing yearwise total employment and mean
fig, ax1 = plt.subplots(figsize=(25, 12))


ax1.bar(yearly_totals['Year'], yearly_totals['emp'], color='skyblue', label='Yearly Total')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Employee Count')
ax1.set_xticks(yearly_totals['Year'])
ax1.set_xticklabels(yearly_totals['Year'])
ax1.set_xticklabels(yearly_totals['Year'], rotation=45)  
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # for the dual axis
ax2.plot(yearly_mean['Year'], yearly_mean['emp'], color='coral', marker='o', label='Yearly Mean', linewidth=2)
ax2.set_ylabel('Yearly Mean Count')


lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')


plt.title('Yearwise Total and Mean Employee Count')
plt.show()


# In[112]:


#IMPUTING THE MISSING VALUES
window_size = 3  
df_imputed = df.copy()

# calculating rolling mean for imputation
rolling_mean = df_imputed['emp'].rolling(window=window_size, min_periods=1).mean()

# imputing missing values 
df_imputed['emp'] = df_imputed['emp'].fillna(rolling_mean) #df_imputed is dataframe after processing


# In[113]:


# Visualizing actual and imputed 
plt.figure(figsize=(10, 6))

plt.plot(df['month'],df['emp'], 'o', label='Original', color='coral') #Actual/Given data
plt.plot(df_imputed['month'],df_imputed['emp'], '-', label='Imputed', color='blue',linewidth=2) #Imputed/processed Data

plt.title('Original and Imputed Time Series')
plt.xlabel('Date')
plt.ylabel('emp')
plt.legend()
plt.show()


# In[114]:


# Visualizing actual and imputed for a small timeframe for better/closer look
start_date = '1990-01-01'
end_date = '1995-01-01'

df_subset = df[(df['month'] >= start_date) & (df['month'] <= end_date)] 
df_subset_imp = df_imputed[(df_imputed['month'] >= start_date) & (df_imputed['month'] <= end_date)]


plt.figure(figsize=(10, 6))
plt.plot(df_subset['month'],df_subset['emp'], '-', label='Original', color='coral',linewidth=1.5) #Actual/Given data
plt.plot(df_subset_imp['month'],df_subset_imp['emp'], 'o', label='Imputed', color='blue',linewidth=1.5) #Imputed/processed Data

plt.title('Original and Imputed Time Series for 5 years data')
plt.xlabel('Date')
plt.ylabel('emp')
plt.legend()
plt.show()


# In[37]:


# Year wise Maximum Employement Months

yearly_max_emp = df_imputed.groupby('Year')['emp'].max()

# visualizing
plt.figure(figsize=(17, 6))
plt.plot(yearly_max_emp.index, yearly_max_emp.values, marker='o', linestyle='-', color='blue', label='Max Employment')

# labels on the line
for year, emp in zip(yearly_max_emp.index, yearly_max_emp.values):
    month = df_imputed.loc[(df_imputed['Year'] == year) & (df_imputed['emp'] == emp), 'month'].iloc[0]
    plt.annotate(month.strftime('%b'), (year, emp), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Maximum Employment Month by Year')
plt.xlabel('Year')
plt.ylabel('Employment')
plt.tight_layout()
plt.show()


# In[38]:


# Year wise Minimum Employement Months
yearly_min_emp = df_imputed.groupby('Year')['emp'].min()

#visualizing
plt.figure(figsize=(17, 6))
plt.plot(yearly_min_emp.index, yearly_min_emp.values, marker='o', linestyle='-', color='red', label='Min Employment')

# labels on the line
for year, emp in zip(yearly_min_emp.index, yearly_min_emp.values):
    month = df_imputed.loc[(df_imputed['Year'] == year) & (df_imputed['emp'] == emp), 'month'].iloc[0]
    plt.annotate(month.strftime('%b'), (year, emp), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Minimum Employment Month by Year')
plt.xlabel('Year')
plt.ylabel('Employment')
plt.legend()

plt.tight_layout()
plt.show()


# In[115]:


#Seasonal Decomposition
result = seasonal_decompose(df_imputed['emp'], model='additive', period=12)

#visualizing trens,seasonality,residual components
plt.rcParams["figure.figsize"] = (10,10)
result.plot()
plt.show()


# In[116]:


#ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(df_imputed['emp'])


# In[117]:


#KPSS Test
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
    result = kpss(timeseries, regression='c')
    print('KPSS Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[3])
    
kpss_test(df_imputed['emp'])


# In[118]:


#ACF and PACF for original data
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF
plt.figure(figsize=(12,6))
plot_acf(df_imputed['emp'], lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()

# PACF
plt.figure(figsize=(6,4))
plot_pacf(df_imputed['emp'], lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.grid(True)
plt.show()


# In[119]:


# performing First Order Differencing to make data stationary
shifted_emp = df_imputed['emp'] - df_imputed['emp'].shift(1)
df1 = pd.DataFrame({'shifted_emp': shifted_emp}) 
df1 = df1.dropna() # important to remove the first row NAN value
df1 = df1.reset_index(drop=True)


# In[120]:


df1.head()


# In[121]:


#ADF test after first order differencing
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(df1['shifted_emp'])


# In[122]:


#ACF and PACF for detrended data, df1 is the name of the dataframe
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF
plt.figure(figsize=(4, 4))
plot_acf(df1['shifted_emp'], lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()

# PACF
plt.figure(figsize=(4,4))
plot_pacf(df1['shifted_emp'], lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.grid(True)
plt.show()


# In[36]:


#can skip to model building, these is extra
#This removes seasonality from the data
#here we are employing this by subtracting the emp value from 12 periods earlier
seasonal_diff = df_imputed['emp'].diff(12)
df_s = pd.DataFrame({'seasonal_diff': seasonal_diff}) 
df_s = df_s.dropna() # important to remove the first row NAN value
df_s = df_s.reset_index(drop=True)


# In[40]:


#ADF test for deseasonalized data
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(df_s['seasonal_diff'])


# In[39]:


#ACF and PACF for deseasonalized data, df_s is the name of the dataframe
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF
plt.figure(figsize=(4, 4))
plot_acf(df_s['seasonal_diff'], lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()

# PACF
plt.figure(figsize=(4,4))
plot_pacf(df_s['seasonal_diff'], lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.grid(True)
plt.show()


# # SARIMA Model

# In[123]:


df1.columns


# In[124]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# 80:20 train and test data, df1 is our dataframe
train_size = int(len(df1) * 0.8)
train, test = df1[:train_size], df1[train_size:]

order = (2, 1, 3) # SARIMA model
seasonal_order = (2, 1, 1,12)  # seasonal order parameters
model = SARIMAX(train['shifted_emp'], order=order, seasonal_order=seasonal_order)
sarima_result = model.fit(disp = False)

forecast = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

mae = mean_absolute_error(test['shifted_emp'], forecast)# Mean Absolute Error (MAE)
print(f' Mean Absolute Error: {mae:.2f}')

rmse = np.sqrt(mean_squared_error(test['shifted_emp'], forecast)) #Root Mean Squared Error (RMSE)
print(f'Root Mean Squared Error : {rmse:.2f}')


# Visualizing
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['shifted_emp'], label='Training Data')
plt.plot(test.index, test['shifted_emp'], label='Testing Data')
plt.plot(test.index, forecast, label='Forecast', color='green')
plt.title('SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Employment')
plt.legend()
plt.show()

# forecasting for the next/following month
last_emp = df_imputed['emp'].iloc[-1] #using the last emp datapoint from processed dataframe
next_month_difference = sarima_result.forecast(steps=1) 
next_month_forecast = last_emp + next_month_difference
print("Forecast for next month:", next_month_forecast)# displays the next month forecasted emp vale


# In[125]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

####
shifted_emp = df_imputed['emp'] - df_imputed['emp'].shift(1)
df1 = pd.DataFrame({'shifted_emp': shifted_emp}) 
df1 = df1.dropna() # important to remove the first row NAN value
df1 = df1.reset_index(drop=True)
df1 = df1.iloc[:-1]
#######

order = (2, 1, 3) # SARIMA model
seasonal_order = (2, 1, 1,12)  # seasonal order parameters
model = SARIMAX(df1['shifted_emp'], order=order, seasonal_order=seasonal_order)
sarima_result = model.fit(disp = False)

mae = mean_absolute_error(test['shifted_emp'], forecast)# Mean Absolute Error (MAE)
print(f' Mean Absolute Error: {mae:.2f}')

rmse = np.sqrt(mean_squared_error(test['shifted_emp'], forecast)) #Root Mean Squared Error (RMSE)
print(f'Root Mean Squared Error : {rmse:.2f}')

# forecasting for the next/following month
last_emp = df_imputed['emp'].iloc[-1] #using the last emp datapoint from processed dataframe
next_month_difference = sarima_result.forecast(steps=1) 
next_month_forecast = last_emp + next_month_difference
print("Forecast for next month:", next_month_forecast)# displays the next month forecasted emp vale
print('Actual given emp:',df_imputed['emp'].iloc[-1] )


# In[ ]:





# In[ ]:





# # Recession Recognition- Early 90's 

# In[33]:


# Considering data around early 90's and assigning to dataframe
start_date = '1985-01-01'
end_date = '1997-01-01'
df_subset = df_imputed[(df_imputed['month'] >= start_date) & (df_imputed['month'] <= end_date)]

# Visualize the data
df_subset.plot(x='month', y='emp', figsize=(12,5), color='coral',grid=True)
plt.grid(alpha=0.5)
plt.xlabel('Date')
plt.ylabel('emp')
plt.title('Employement Data around 1990')
plt.show()


# In[34]:


#emp during recession
start_date = '1990-01-01'
end_date = '1993-06-01'
df_subset = df_imputed[(df_imputed['month'] >= start_date) & (df_imputed['month'] <= end_date)]

# Visualizing
df_subset.plot(x='month', y='emp', figsize=(12,5), color='red',grid=True)
plt.grid(alpha=0.5)
plt.xlabel('Date')
plt.ylabel('emp')
plt.title('Employement Data around Recession Pattern')
plt.show()


# In[41]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

start_date = '1978-02-01'
end_date = '1990-12-01'
df_subset = df_imputed[(df_imputed['month'] >= start_date) & (df_imputed['month'] <= end_date)]

shifted_emp = df_subset['emp'] - df_subset['emp'].shift(1)
df1 = pd.DataFrame({'shifted_emp': shifted_emp}) 
df1 = df1.dropna() # important to remove the first row NAN value
df1 = df1.reset_index(drop=True)

# SARIMA model
order = (2, 1, 3)  # order parameters
seasonal_order = (2, 1, 1, 12)  # seasonal order parameters
model = SARIMAX(train['shifted_emp'], order=order, seasonal_order=seasonal_order)
sarima_result = model.fit(disp = False)

forecast_diff = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, typ='linear')

# forecasting for the next/following 36 months
last_emp = df_subset['emp'].iloc[-1] #using the last emp datapoint from processed dataframe
next_month_difference = sarima_result.forecast(steps=36) 
next_month_forecast = [last_emp + diff for diff in next_month_difference]
#print("forecasted_emp:", next_month_forecast)# displays the next month forecasted emp vale
forecast_df = pd.DataFrame({'forecasted_emp': next_month_forecast})
#print(forecast_df.head())

start_date_1993 = '1993-01-01'
end_date_1993 = '1993-12-01' 

df_subset1 = df_imputed[(df_imputed['month'] >= start_date_1993) & (df_imputed['month'] <= end_date_1993)]
df_subset1 = df_subset1.reset_index(drop=True)
#print(df_subset1)

# Merging the two DataFrames 
joined_df = pd.concat([forecast_df.iloc[-12:].reset_index(drop=True), df_subset1.reset_index(drop=True)], axis=1)
joined_df['diff'] = joined_df['emp'] - joined_df['forecasted_emp']
joined_df = joined_df[['month','Year','emp','forecasted_emp', 'diff']]


print(joined_df)


# In[42]:


plt.figure(figsize=(12, 6))

# Plot actual employment
plt.plot(joined_df['month'], joined_df['emp'], marker='o', linestyle='-', color='blue', label='Actual Employment')

# Plot forecasted employment
plt.plot(joined_df['month'], joined_df['forecasted_emp'], marker='o', linestyle='-', color='red', label='Forecasted Employment')

# Set plot title and labels
plt.title('Actual vs Forecasted Employment')
plt.xlabel('Month')
plt.ylabel('Employment')
plt.title('Actual vs Forecasted Employment')
plt.legend()

# Show plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[ ]:




