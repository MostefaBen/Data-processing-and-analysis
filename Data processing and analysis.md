# Data Preprocessing


```python
# import libraries
import torch
import os
import time
import timeit
import numpy as np
from os import path
import pandas as pd
```


```python
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
```


```python
# checking  if the CSV file exists 
data_file = os.path.join('data', 'houses.csv')
if path.exists(data_file):
    print("True")
```

    True
    


```python
# Reading a CSV file 
data_file = os.path.join('data', 'houses.csv')
if path.exists(data_file):
    data = pd.read_csv(data_file)
    print(data)
```

      Country  Rooms  Price  Size
    0     USA    NaN  12500   200
    1     NaN    3.0  16000   150
    2  France    4.0  17800   170
    3     NaN    NaN  14000    90
    


```python
# Data Frame indexing and slicing
# Indexing specific rows and columns
print(data.loc[[1,3],['Country', 'Price']])
```

      Country  Price
    1     NaN  16000
    3     NaN  14000
    


```python
# Column indexing
data.loc[:,['Price']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Row indexing
data.loc[0]
```




    Country      USA
    Rooms        NaN
    Price      12500
    Size         200
    Name: 0, dtype: object




```python
# More about indexing
data.Country[0], data.Price[0]
```




    ('USA', 12500)




```python
# Slicing rows and columns
print(data.loc[0:2,['Country', 'Price']])
```

      Country  Price
    0     USA  12500
    1     NaN  16000
    2  France  17800
    


```python
# Data Frame Type
data.dtypes
```




    Country     object
    Rooms      float64
    Price        int64
    Size         int64
    dtype: object




```python
# Get column type
data['Price'].dtype
```




    dtype('int64')



# Data Analysis


```python
# Describing Data Frame 
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Price</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.500000</td>
      <td>15075.000000</td>
      <td>152.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.707107</td>
      <td>2314.267343</td>
      <td>46.457866</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>12500.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.250000</td>
      <td>13625.000000</td>
      <td>135.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.500000</td>
      <td>15000.000000</td>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.750000</td>
      <td>16450.000000</td>
      <td>177.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>17800.000000</td>
      <td>200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get max value of each column
data.describe().max()
```




    Rooms        4.0
    Price    17800.0
    Size       200.0
    dtype: float64



##### Memory saving
As we can see, the maximum value is 17800.0, which is less than the maximum value of uint16 (65535), so all we need in terms of memory is of type uint16


```python
# Lets compute the amounts of used memory
data.info(memory_usage='deep'), data.memory_usage(deep=True).sum()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4 entries, 0 to 3
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   Country  2 non-null      object 
     1   Rooms    2 non-null      float64
     2   Price    4 non-null      int64  
     3   Size     4 non-null      int64  
    dtypes: float64(1), int64(2), object(1)
    memory usage: 411.0 bytes
    




    (None, 411)




```python
# Check if Price column is numeric
np.issubdtype(data['Price'].dtype, np.number)
```




    True




```python
# Get the name of each column
data.columns
```




    Index(['Country', 'Rooms', 'Price', 'Size'], dtype='object')




```python
# Convert pandas indexes to python list
data.columns.tolist()
```




    ['Country', 'Rooms', 'Price', 'Size']




```python
# Before converting Data Frame Type to unit16
data.dtypes
```




    Country     object
    Rooms      float64
    Price        int64
    Size         int64
    dtype: object




```python
# Building a dictionary comprehension
start_time = time.time()
temp_cols_uint16 = {col: np.uint16 if np.issubdtype(data[col].dtype, np.number) else data[col].dtype for col in data.columns}
print(temp_cols_uint16)
print("Processing time: ", time.time()-start_time)
```

    {'Country': dtype('O'), 'Rooms': <class 'numpy.uint16'>, 'Price': <class 'numpy.uint16'>, 'Size': <class 'numpy.uint16'>}
    Processing time:  0.0009999275207519531
    


```python
# Vectorize the check of Data Frame columns
start_time = time.time()
is_number = np.vectorize(lambda x: np.issubdtype(x, np.number) )
print(is_number(data.dtypes))
print("Processing time: ", time.time()-start_time)
```

    [False  True  True  True]
    Processing time:  0.002000093460083008
    


```python
# Convert country column into string
# In Pandas string is represented as a Python Object
data['Country'] = data['Country'] .astype(str)
data.dtypes
```




    Country     object
    Rooms      float64
    Price        int64
    Size         int64
    dtype: object




```python
#start_time = time.time()
#is_number = np.vectorize(lambda x, y: {np.uint16, x} if np.issubdtype(y, np.number) else {})
#print(is_number(data.columns, data.dtypes))
#print("Processing time: ", time.time()-start_time)
```


```python
# Replace Nan with 0
tmp_data = data.fillna(0)
tmp_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Rooms</th>
      <th>Price</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>USA</td>
      <td>0.0</td>
      <td>12500</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nan</td>
      <td>3.0</td>
      <td>16000</td>
      <td>150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>France</td>
      <td>4.0</td>
      <td>17800</td>
      <td>170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nan</td>
      <td>0.0</td>
      <td>14000</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
# After converting numerical columns into uint16
data = tmp_data.astype(temp_cols_uint16)
data.dtypes  
```




    Country    object
    Rooms      uint16
    Price      uint16
    Size       uint16
    dtype: object




```python
# To get the mostly-accurate memory usage after convertion to uint16
data.info(memory_usage='deep'), data.memory_usage(deep=True).sum()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4 entries, 0 to 3
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   Country  4 non-null      object
     1   Rooms    4 non-null      uint16
     2   Price    4 non-null      uint16
     3   Size     4 non-null      uint16
    dtypes: object(1), uint16(3)
    memory usage: 395.0 bytes
    




    (None, 395)



#### By converting only Data Frame columns to uint16, we have optimized the memory by 3.89 %


```python
# Changing the position of a column (method 1)
temp_cols = data.columns.tolist()
temp_cols =  temp_cols[:-2] + temp_cols[-1:]  + temp_cols[-2:-1] 
temp_cols
```




    ['Country', 'Rooms', 'Size', 'Price']




```python
# Changing the position of a column (method 2)
#col = data.pop('Size')
#data = data.insert(3, 'Size', col)
#data
```


```python
# The new Data Frame
data = data[temp_cols]
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Rooms</th>
      <th>Size</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>USA</td>
      <td>0</td>
      <td>200</td>
      <td>12500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nan</td>
      <td>3</td>
      <td>150</td>
      <td>16000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>France</td>
      <td>4</td>
      <td>170</td>
      <td>17800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nan</td>
      <td>0</td>
      <td>90</td>
      <td>14000</td>
    </tr>
  </tbody>
</table>
</div>



# Data Preparation


```python
# we use iloc to select columns
inputs, targets = data.iloc[:, 0:3], data.iloc[:, 3]
print(inputs,"\n")
print(targets)
```

      Country  Rooms  Size
    0     USA      0   200
    1     nan      3   150
    2  France      4   170
    3     nan      0    90 
    
    0    12500
    1    16000
    2    17800
    3    14000
    Name: Price, dtype: uint16
    


```python
# transform pandas tables into numpy matrices
inputs.values, targets.values
```




    (array([['USA', 0, 200],
            ['nan', 3, 150],
            ['France', 4, 170],
            ['nan', 0, 90]], dtype=object),
     array([12500, 16000, 17800, 14000], dtype=uint16))




```python
# dealing with missing data
# in this example, we will apply imputation heuristics. For categorical input fields, we can treat NaN as a category.
# dummy_na=True shows the column of the Nan
inputs = pd.get_dummies(inputs, dummy_na=True) 
inputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Size</th>
      <th>Country_France</th>
      <th>Country_USA</th>
      <th>Country_nan</th>
      <th>Country_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>200</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>170</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dealing with missing data
# in this example, we replace the NaN entries with the mean value of the corresponding column.
# we use fillna to fill the Nan entry
inputs =  inputs.fillna(inputs.mean())
inputs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rooms</th>
      <th>Size</th>
      <th>Country_France</th>
      <th>Country_USA</th>
      <th>Country_nan</th>
      <th>Country_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>200</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>170</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Conversion to the Tensor Format


```python
# Data frame type
data.dtypes
```




    Country    object
    Rooms      uint16
    Size       uint16
    Price      uint16
    dtype: object



Pytorch, for now, does not support unsigned 16-bit integer.
The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.


```python
# Converting data frame to signed 32-bit integer (int32)
inputs, targets = inputs.astype(np.int32), targets.astype(np.int32)
inputs.dtypes, targets.dtypes
```




    (Rooms             int32
     Size              int32
     Country_France    int32
     Country_USA       int32
     Country_nan       int32
     Country_nan       int32
     dtype: object,
     dtype('int32'))




```python
# Convertng data frame to tensor
X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
X, y
```




    (tensor([[  0, 200,   0,   1,   0,   0],
             [  3, 150,   0,   0,   1,   0],
             [  4, 170,   1,   0,   0,   0],
             [  0,  90,   0,   0,   1,   0]], dtype=torch.int32),
     tensor([12500, 16000, 17800, 14000], dtype=torch.int32))


