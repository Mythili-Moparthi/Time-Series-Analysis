#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


# # USA data analysis - GDP,PPI,Crudeoil,RealEstate, Employement

# In[44]:


emp = pd.read_csv("/Users/mythilimoparthi/Desktop/Notseasonallyadjusted_emp.csv")
start_date = '1986-01-01'
end_date = '2020-12-01'
df_emp = emp[(emp['DATE'] >= start_date) & (emp['DATE'] <= end_date)]
df_emp = df_emp.rename(columns={'LNU02000000': 'emp'})
df_emp.head()


# In[45]:


gdp = pd.read_csv("/Users/mythilimoparthi/Desktop/GDP.csv")
start_date = '1986-01-01'
end_date = '2020-12-01'
df_gdp = gdp[(gdp['DATE'] >= start_date) & (gdp['DATE'] <= end_date)]
df_gdp = df_gdp.rename(columns={'GDP': 'gdp'})
df_gdp.head()


# In[46]:


#producer price Index by commodity for all commodities
ppi = pd.read_csv("/Users/mythilimoparthi/Desktop/PPIACO.csv")
start_date = '1986-01-01'
end_date = '2020-12-01'
df_ppi = ppi[(ppi['DATE'] >= start_date) & (ppi['DATE'] <= end_date)]
df_ppi = df_ppi.rename(columns={'PPIACO': 'ppi'})
df_ppi.head()


# In[47]:


oil = pd.read_csv("/Users/mythilimoparthi/Desktop/MCOILWTICO.csv")
start_date = '1986-01-01'
end_date = '2020-12-01'
df_oil = oil[(oil ['DATE'] >= start_date) & (oil ['DATE'] <= end_date)]
df_oil = df_oil .rename(columns={'MCOILWTICO': 'oil'})
df_oil.head()


# In[48]:


rest = pd.read_csv("/Users/mythilimoparthi/Desktop/MSPUS.csv")
start_date = '1986-01-01'
end_date = '2020-12-01'
df_rest = rest [(rest ['DATE'] >= start_date) & (rest  ['DATE'] <= end_date)]
df_rest = df_rest .rename(columns={'MSPUS': 'rest'})
df_rest .head()


# In[49]:


merged_df = pd.merge(df_emp, df_gdp, on='DATE')


# In[50]:


merged_df = pd.merge(merged_df, df_rest, on='DATE')


# In[51]:


merged_df = pd.merge(merged_df, df_oil, on='DATE')


# In[52]:


merged_df = pd.merge(merged_df, df_ppi, on='DATE')


# In[53]:


merged_df['DATE'] = pd.to_datetime(merged_df['DATE'])
merged_df['Year'] = merged_df['DATE'].dt.year


# In[54]:


merged_df.head()


# In[61]:


fig, ax1 = plt.subplots(figsize=(20, 6))

#emp and gdp on one left axis
ax1.set_xlabel('Year')
ax1.set_ylabel('Value', color='black')
ax1.plot(merged_df['Year'], merged_df['emp'], linestyle='-', color='blue', label='Employment')
ax1.plot(merged_df['Year'], merged_df['gdp'], linestyle='-', color='coral', label='GDP')
ax1.plot(merged_df['Year'], merged_df['rest'], linestyle='-', color='red', label='RealEstate')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()  

# realestate,ppi,oil on right axis
ax2.set_ylabel('Value', color='black')

ax2.plot(merged_df['Year'], merged_df['ppi'], linestyle='-', color='green', label='PPI')
ax2.tick_params(axis='y', labelcolor='green')

ax2.plot(merged_df['Year'], merged_df['oil'], linestyle='-', color='black', label='CrudeOil')
ax2.tick_params(axis='y', labelcolor='black')
fig.suptitle('Line Chart')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xticks(rotation=45)
plt.show()


# In[56]:


plt.figure(figsize=(20, 6))
plt.plot(merged_df['Year'], merged_df['oil'], linestyle='-', color='black', label='CrudeOil')
plt.plot(merged_df['Year'], merged_df['ppi'], linestyle='-', color='blue', label='PPI')
plt.title('Line Chart')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[57]:


fig, ax1 = plt.subplots(figsize=(20, 6))

#emp and realestate on one left axis
ax1.set_xlabel('Year')
ax1.set_ylabel('Value', color='black')
ax1.plot(merged_df['Year'], merged_df['emp'], linestyle='-', color='blue', label='Employment')
ax1.plot(merged_df['Year'], merged_df['rest'], linestyle='-', color='red', label='RealEstate')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()  

# gdp on right axis
ax2.set_ylabel('Value', color='black')

ax2.plot(merged_df['Year'], merged_df['gdp'], linestyle='-', color='coral', label='GDP')
ax2.tick_params(axis='y', labelcolor='black')

fig.suptitle('Line Chart')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xticks(rotation=45)
plt.show()


# In[58]:


plt.figure(figsize=(20, 6))

plt.plot(merged_df['Year'], merged_df['gdp'], linestyle='-', color='green', label='GDP')

plt.title('Line Chart')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[59]:


plt.figure(figsize=(12, 8))

# histograms for each variable
plt.subplot(3, 2, 1)
sns.histplot(data=merged_df, x='emp', kde=True, color='skyblue', bins=30)
plt.title('Employment')

plt.subplot(3, 2, 2)
sns.histplot(data=merged_df, x='gdp', kde=True, color='salmon', bins= 30)
plt.title('GDP')

plt.subplot(3, 2, 3)
sns.histplot(data=merged_df, x='oil', kde=True, color='green', bins=30)
plt.title('Crude Oil')

plt.subplot(3, 2, 4)
sns.histplot(data=merged_df, x='rest', kde=True, color='orange', bins=30)
plt.title('Real Estate')

plt.subplot(3, 2, 5)
sns.histplot(data=merged_df, x='ppi', kde=True, color='purple', bins=30)
plt.title('PPI Index')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




