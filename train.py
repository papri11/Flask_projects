import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold 
import joblib


house = pd.read_csv("Maison.csv")
house = house.rename(index = str, columns = {'PRIX':'price','SUPERFICIE': 'area','CHAMBRES': 'rooms', 
                         'SDB': 'bathroom', 'ETAGES': 'floors','ALLEE': 'driveway',
                         'SALLEJEU':'game_room', 'CAVE': 'cellar', 
                         'GAZ': 'gas', 'AIR':'air', 'GARAGES': 'garage', 'SITUATION': 'situation'})

house.duplicated().sum()
house.drop_duplicates()


x=house.drop(['price'],axis=1)
y=house[['price']]
y=y.values.reshape(-1,1)

num=x.select_dtypes(include='number')

def unique_levels(x):
    x=x.value_counts().count()
    return(x)

df_value_counts=pd.DataFrame(num.apply(lambda x : unique_levels(x)))
df_value_counts.columns=['feature_levels']
df_value_counts.head()

slice1=df_value_counts.loc[df_value_counts['feature_levels']<=20]
cat_list=slice1.index
cat=num.loc[:,cat_list]

cat['rooms']=cat['rooms'].astype('object')
cat['bathroom']=cat['bathroom'].astype('object')
cat['floors']=cat['floors'].astype('object')
cat['driveway']=cat['driveway'].astype('object')
cat['game_room']=cat['game_room'].astype('object')
cat['cellar']=cat['cellar'].astype('object')
cat['gas']=cat['gas'].astype('object')
cat['air']=cat['air'].astype('object')
cat['garage']=cat['garage'].astype('object')
cat['situation']=cat['situation'].astype('object')


slice2=df_value_counts.loc[df_value_counts['feature_levels']>20]
num_list=slice2.index
num=num.loc[:,num_list]

def outlier_cap(x):
    x=x.clip(lower=x.quantile(0.01))
    x=x.clip(upper=x.quantile(0.99))
    return(x)

num=num.apply(lambda x : outlier_cap(x))

char = cat.loc[:,cat.apply(pd.Series.nunique) != 1]

X_all=pd.concat([char,num],axis=1,join="inner")

sc_X = StandardScaler()
sc_X = sc_X.fit_transform(X_all)
#Convert to table format - StandardScaler 
sc_X = pd.DataFrame(data=sc_X, columns=['rooms','bathroom', 'floors','driveway', 'game_room','cellar',
        'gas', 'air', 'garage', 'situation','area'])




X_train, X_test, y_train, y_test = train_test_split( sc_X, y, test_size=0.3, random_state=42)

print("Shape of Training Data",X_train.shape)
print("Shape of Testing Data",X_test.shape)
print("Attrition Rate in Training Data",y_train.mean())
print("Attrition Rate in Testing Data",y_test.mean())


lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
print("Accuracy --> ", lm.score(X_test, y_test)*100)


print('model is running successfully')

joblib.dump(lm ,'dib_lr.pkl')
