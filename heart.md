# 心跳信号分类预测
## 数据集： https://tianchi.aliyun.com/competition/entrance/531883/information
加载分析所用的python库


```python
#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')
#import missingno as msno
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
```

# 一、数据获取及预处理

## 1.导入训练集与测试集，并观察首尾信息及数据集大小

读取数据集


```python
Train_data = pd.read_csv('E:/data mining/train.csv')
Test_data = pd.read_csv('E:/data mining/testA.csv')
```

使用head、tail函数查看首尾信息


```python
#观察训练集首尾数据
Train_data.head().append(Train_data.tail())
# 数据集的含义
# 列名----------含义
# id    为心跳信号分配的唯一标识
# heartbeat_signals 心跳信号序列
# label 心跳信号类别（0、1、2、3）
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
      <th>id</th>
      <th>heartbeat_signals</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.9912297987616655,0.9435330436439665,0.764677...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.9714822034884503,0.9289687459588268,0.572932...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.0,0.9591487564065292,0.7013782792997189,0.23...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.9757952826275774,0.9340884687738161,0.659636...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0,0.055816398940721094,0.26129357194994196,0...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>99995</td>
      <td>1.0,0.677705342021188,0.22239242747868546,0.25...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>99996</td>
      <td>0.9268571578157265,0.9063471198026871,0.636993...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>99997</td>
      <td>0.9258351628306013,0.5873839035878395,0.633226...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>99998</td>
      <td>1.0,0.9947621698382489,0.8297017704865509,0.45...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>99999</td>
      <td>0.9259994004527861,0.916476635326053,0.4042900...</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#观察测试集首尾数据
Test_data.head().append(Test_data.tail())
# 数据集的含义
# 列名----------含义
# id    为心跳信号分配的唯一标识
# heartbeat_signals 心跳信号序列
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
      <th>id</th>
      <th>heartbeat_signals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100000</td>
      <td>0.9915713654170097,1.0,0.6318163407681274,0.13...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>0.6075533139615096,0.5417083883163654,0.340694...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>0.9752726292239277,0.6710965234906665,0.686758...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100003</td>
      <td>0.9956348033996116,0.9170249621481004,0.521096...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100004</td>
      <td>1.0,0.8879490481178918,0.745564725322326,0.531...</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>119995</td>
      <td>1.0,0.8330283177934747,0.6340472606311671,0.63...</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>119996</td>
      <td>1.0,0.8259705825857048,0.4521053488322387,0.08...</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>119997</td>
      <td>0.951744840752379,0.9162611283848351,0.6675251...</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>119998</td>
      <td>0.9276692903808186,0.6771898159607004,0.242906...</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>119999</td>
      <td>0.6653212231837624,0.527064114047737,0.5166625...</td>
    </tr>
  </tbody>
</table>
</div>



使用shape函数查看数据集大小


```python
#观察训练集数据集大小
Train_data.shape 
```




    (100000, 3)




```python
#观察测试集数据集大小
Test_data.shape
```




    (20000, 2)



## 2.查看数据集统计量、数据类型

descirbe函数查看各列的统计量，查看数据的大致范围，同时可以根据最大最小值判断是否含有特殊值。
info函数查看各列的数据类型，判断是否有异常数据


```python
Train_data.describe()
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
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100000.000000</td>
      <td>100000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>49999.500000</td>
      <td>0.856960</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28867.657797</td>
      <td>1.217084</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24999.750000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49999.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>74999.250000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99999.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
Train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 3 columns):
     #   Column             Non-Null Count   Dtype  
    ---  ------             --------------   -----  
     0   id                 100000 non-null  int64  
     1   heartbeat_signals  100000 non-null  object 
     2   label              100000 non-null  float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 2.3+ MB
    


```python
Test_data.describe()
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
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>109999.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5773.647028</td>
    </tr>
    <tr>
      <th>min</th>
      <td>100000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>104999.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>109999.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>114999.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>119999.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
Test_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20000 entries, 0 to 19999
    Data columns (total 2 columns):
     #   Column             Non-Null Count  Dtype 
    ---  ------             --------------  ----- 
     0   id                 20000 non-null  int64 
     1   heartbeat_signals  20000 non-null  object
    dtypes: int64(1), object(1)
    memory usage: 312.6+ KB
    

## 3.判断异常值、缺失值

使用isnull函数查看每列是否存在nan


```python
Train_data.isnull().sum()
```




    id                   0
    heartbeat_signals    0
    label                0
    dtype: int64




```python
Test_data.isnull().sum()
```




    id                   0
    heartbeat_signals    0
    dtype: int64



可见，训练集和测试集均不存在缺失值，非常理想

# 二、数据分析与可视化

## 1.查看数据的总体分布情况
使用 sns.distplot() 方法 绘制分布直方图：

代码使用了seaborn库，Seaborn是基于matplotlib的Python可视化库。 它提供了一个高级界面来绘制有吸引力的统计图形。Seaborn其实是在matplotlib的基础上进行了更高级的API封装。

核密度估计（kernel density estimation）是在概率论中用来估计未知的密度函数，属于非参数检验方法之一。核密度估计方法不利用有关数据分布的先验知识，对数据分布不附加任何假定，是一种从数据样本本身出发研究数据分布特征的方法

seaborn库的displot()函数集合了matplotlib的hist()#直方图函数与核函数估计kdeplot的功能，增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途。具体用法如下：

seaborn.distplot(a,bins=None,hist=True,kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)

通过hist和kde参数调节是否显示直方图及核密度估计(默认hist,kde均为True)
fit参数控制拟合的参数分布图形，能够直观地评估它与观察数据的对应关系



```python
import scipy.stats as st
#图中的曲线部分即为估计的概率分布
y = Train_data['label']
plt.figure(1); plt.title('Default')
sns.distplot(y, rug=True, bins=20)
plt.figure(2); plt.title('Normal')
#fit=norm拟合表准的正态分布（即图中黑色线）
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
#fit=lognorm拟合对数正态分布（即图中黑色线）
sns.distplot(y, kde=False, fit=st.lognorm)

```




    <AxesSubplot:title={'center':'Log Normal'}, xlabel='label'>




    
![png](output_23_1.png)
    



    
![png](output_23_2.png)
    



    
![png](output_23_3.png)
    


## 2.查看skewness and kurtosis
使用 skew() 方法查看 偏度 (skewness)， 使用 kurt() 方法查看 峰度 (kurtosis)：

偏度（skewness），是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）。（衡量偏离正态的程度）

峰度（peakedness；kurtosis）又称峰态系数。表征概率密度分布曲线在平均值处峰值高低的特征数。直观看来，峰度反映了峰部的尖度。
一般地：正态分布的峰度（系数）为常数3，均匀分布的峰度（系数）为常数1.8



```python
sns.distplot(Train_data['label']);
print("Skewness: %f" % Train_data['label'].skew())
print("Kurtosis: %f" % Train_data['label'].kurt())
```

    Skewness: 0.871005
    Kurtosis: -1.009573
    


    
![png](output_26_1.png)
    


综上可知，偏度 (skewness) 大于 0，表示数据分布倾向于右偏，长尾在右；峰度 (kurtosis) 小于 0，表示数据分布 与 正态分布相比，较为平坦，为平顶峰。

调用 skew() 和 kurt() 方法分别查看偏度和峰度


```python
Train_data.skew(), Train_data.kurt()
```




    (id       0.000000
     label    0.871005
     dtype: float64,
     id      -1.200000
     label   -1.009573
     dtype: float64)



绘制偏度直方图


```python
sns.distplot(Train_data.skew(), color='green', axlabel ='Skewness')
```




    <AxesSubplot:xlabel='Skewness', ylabel='Density'>




    
![png](output_31_1.png)
    


绘制峰度直方图


```python
sns.distplot(Train_data.kurt(), color='purple', axlabel ='Kurtness')
```




    <AxesSubplot:xlabel='Kurtness', ylabel='Density'>




    
![png](output_33_1.png)
    


## 3.查看预测值频数
使用 value_counts() 方法 统计各类别数：


```python
Train_data['label'].value_counts()
```




    0.0    64327
    3.0    17912
    2.0    14199
    1.0     3562
    Name: label, dtype: int64




```python
plt.hist(Train_data['label'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()

```


    
![png](output_36_0.png)
    


可见，训练集的类别存在类别不均衡问题。
