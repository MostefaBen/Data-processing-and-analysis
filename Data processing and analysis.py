#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[1]:


# import libraries
import torch
import os
import time
import timeit
import numpy as np
from os import path
import pandas as pd


# In[2]:


# Creaing a folder
os.makedirs(os.path.join('data'), exist_ok=True)
# Creating (Writing inside) a CSV file 
data_file = os.path.join('data', 'houses.csv')
with open(data_file, 'w') as f:
     f.write('''Country,Rooms,Price,Size
USA,NA,12500,200
NA,3,16000,150
France,4,17800,170
NA,NA,14000,90''')


# In[3]:


# checking  if the CSV file exists 
data_file = os.path.join('data', 'houses.csv')
if path.exists(data_file):
    print("True")


# In[4]:


# Reading a CSV file 
data_file = os.path.join('data', 'houses.csv')
if path.exists(data_file):
    data = pd.read_csv(data_file)
    print(data)


# In[5]:


# Data Frame indexing and slicing
# Indexing specific rows and columns
print(data.loc[[1,3],['Country', 'Price']])


# In[6]:


# Column indexing
data.loc[:,['Price']]


# In[7]:


# Row indexing
data.loc[0]


# In[8]:


# More about indexing
data.Country[0], data.Price[0]


# In[9]:


# Slicing rows and columns
print(data.loc[0:2,['Country', 'Price']])


# In[10]:


# Data Frame Type
data.dtypes


# In[11]:


# Get column type
data['Price'].dtype


# # Data Analysis

# In[12]:


# Describing Data Frame 
data.describe()


# In[13]:


# Get max value of each column
data.describe().max()


# ##### Memory saving
# As we can see, the maximum value is 17800.0, which is less than the maximum value of uint16 (65535), so all we need in terms of memory is of type uint16

# In[14]:


# Lets compute the amounts of used memory
data.info(memory_usage='deep'), data.memory_usage(deep=True).sum()


# In[15]:


# Check if Price column is numeric
np.issubdtype(data['Price'].dtype, np.number)


# In[16]:


# Get the name of each column
data.columns


# In[17]:


# Convert pandas indexes to python list
data.columns.tolist()


# In[18]:


# Before converting Data Frame Type to unit16
data.dtypes


# In[19]:


# Building a dictionary comprehension
start_time = time.time()
temp_cols_uint16 = {col: np.uint16 if np.issubdtype(data[col].dtype, np.number) else data[col].dtype for col in data.columns}
print(temp_cols_uint16)
print("Processing time: ", time.time()-start_time)


# In[20]:


# Vectorize the check of Data Frame columns
start_time = time.time()
is_number = np.vectorize(lambda x: np.issubdtype(x, np.number) )
print(is_number(data.dtypes))
print("Processing time: ", time.time()-start_time)


# In[21]:


# Convert country column into string
# In Pandas string is represented as a Python Object
data['Country'] = data['Country'] .astype(str)
data.dtypes


# In[22]:


#start_time = time.time()
#is_number = np.vectorize(lambda x, y: {np.uint16, x} if np.issubdtype(y, np.number) else {})
#print(is_number(data.columns, data.dtypes))
#print("Processing time: ", time.time()-start_time)


# In[23]:


# Replace Nan with 0
tmp_data = data.fillna(0)
tmp_data


# In[24]:


# After converting numerical columns into uint16
data = tmp_data.astype(temp_cols_uint16)
data.dtypes  


# In[25]:


# To get the mostly-accurate memory usage after convertion to uint16
data.info(memory_usage='deep'), data.memory_usage(deep=True).sum()


# #### By converting only Data Frame columns to uint16, we have optimized the memory by 3.89 %

# In[26]:


# Changing the position of a column (method 1)
temp_cols = data.columns.tolist()
temp_cols =  temp_cols[:-2] + temp_cols[-1:]  + temp_cols[-2:-1] 
temp_cols


# In[27]:


# Changing the position of a column (method 2)
#col = data.pop('Size')
#data = data.insert(3, 'Size', col)
#data


# In[28]:


# The new Data Frame
data = data[temp_cols]
data


# # Data Preparation

# In[29]:


# we use iloc to select columns
inputs, targets = data.iloc[:, 0:3], data.iloc[:, 3]
print(inputs,"\n")
print(targets)


# In[30]:


# transform pandas tables into numpy matrices
inputs.values, targets.values


# In[31]:


# dealing with missing data
# in this example, we will apply imputation heuristics. For categorical input fields, we can treat NaN as a category.
# dummy_na=True shows the column of the Nan
inputs = pd.get_dummies(inputs, dummy_na=True) 
inputs


# In[32]:


# dealing with missing data
# in this example, we replace the NaN entries with the mean value of the corresponding column.
# we use fillna to fill the Nan entry
inputs =  inputs.fillna(inputs.mean())
inputs


# # Conversion to the Tensor Format

# In[36]:


# Data frame type
data.dtypes


# Pytorch, for now, does not support unsigned 16-bit integer.
# The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.

# In[34]:


# Converting data frame to signed 32-bit integer (int32)
inputs, targets = inputs.astype(np.int32), targets.astype(np.int32)
inputs.dtypes, targets.dtypes


# In[35]:


# Convertng data frame to tensor
X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
X, y

