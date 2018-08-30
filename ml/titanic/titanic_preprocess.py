import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/TeamLab/machine_learning_from_scratch_with_python/master/code/ch12/titanic/train.csv')

print(df.shape) #(891, 12)
print(df.head())
'''
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S 
'''

#####불필요한 독립변수 제거

del df['PassengerId']
del df['Name']
del df['Ticket']
del df['Cabin']

print(df.head())
'''
   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked
0         0       3    male  22.0      1      0   7.2500        S
1         1       1  female  38.0      1      0  71.2833        C
2         1       3  female  26.0      0      0   7.9250        S
3         1       1  female  35.0      1      0  53.1000        S
4         0       3    male  35.0      0      0   8.0500        S
'''

#####값 모르는 독립변수 제거

print(df.isnull())
print(df.isnull().sum())
'''
Survived      0
Pclass        0
Sex           0
Age         177
SibSp         0
Parch         0
Fare          0
Embarked      2
dtype: int64
'''

#df=df.dropna()
#df=df.dropna(how='all')
#df=df.dropna(thresh=3)
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

print(df.isnull())
print(df.isnull().sum())
'''
Survived    0
Pclass      0
Sex         0
Age         0
SibSp       0
Parch       0
Fare        0
Embarked    0
dtype: int64
'''

#####범주형 독립변수 더미 변수로 변환

print(df.head())
'''
   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked
0         0       3    male  22.0      1      0   7.2500        S
1         1       1  female  38.0      1      0  71.2833        C
2         1       3  female  26.0      0      0   7.9250        S
3         1       1  female  35.0      1      0  53.1000        S
4         0       3    male  35.0      0      0   8.0500        S
'''
print(df['Pclass'].unique()) #[3 1 2]
print(df['Sex'].unique()) #['male' 'female']
print(df['Embarked'].unique()) #['S' 'C' 'Q']

print(df.dtypes)
'''
Survived      int64
Pclass        int64
Sex          object
Age         float64
SibSp         int64
Parch         int64
Fare        float64
Embarked     object
dtype: object
'''

df["Pclass"] = df["Pclass"].astype("category")
df["Sex"] = df["Sex"].astype("category")
#df['Sex']=df['Sex'].replace(['female','male'],[0,1])
df["Embarked"] = df["Embarked"].astype("category")

print(df.dtypes)
'''
Survived       int64
Pclass      category
Sex         category
Age          float64
SibSp          int64
Parch          int64
Fare         float64
Embarked    category
dtype: object
'''

df=pd.get_dummies(df)
print(df.head())
'''
   Survived   Age  SibSp  Parch     Fare  Pclass_1  Pclass_2  Pclass_3  \
0         0  22.0      1      0   7.2500         0         0         1   
1         1  38.0      1      0  71.2833         1         0         0   
2         1  26.0      0      0   7.9250         0         0         1   
3         1  35.0      1      0  53.1000         1         0         0   
4         0  35.0      0      0   8.0500         0         0         1   

   Sex_female  Sex_male  Embarked_C  Embarked_Q  Embarked_S  
0           0         1           0           0           1  
1           1         0           1           0           0  
2           1         0           0           0           1  
3           1         0           0           0           1  
4           0         1           0           0           1
'''

#####독립변수와 종속변수 구분 

x_data=df.iloc[:,1:]
x_data=x_data.values
y_data=df['Survived']
#y_data=df.iloc[:, 0]
y_data=y_data.values
print(x_data.shape) #
print(y_data.shape) #
print(x_data[:5])
'''
[[22.      1.      0.      7.25    0.      0.      1.      0.      1.
   0.      0.      1.    ]
 [38.      1.      0.     71.2833  1.      0.      0.      1.      0.
   1.      0.      0.    ]
 [26.      0.      0.      7.925   0.      0.      1.      1.      0.
   0.      0.      1.    ]
 [35.      1.      0.     53.1     1.      0.      0.      1.      0.
   0.      0.      1.    ]
 [35.      0.      0.      8.05    0.      0.      1.      0.      1.
   0.      0.      1.    ]]
'''
print(y_data[:5]) #[0 1 1 1 0]

np.save('titanic_x_data.npy', x_data)
np.save('titanic_y_data.npy', y_data)
