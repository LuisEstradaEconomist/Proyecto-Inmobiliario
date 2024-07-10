import pandas as pd 
import numpy as np
import seaborn as sns
import statsmodels
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor



data = pd.read_csv("C:/Users/luispe/OneDrive - Corporación Interactuar/Aleatorio/Documentos Personales/Proyecto Inmobiliario/datos_fincaraiz.csv", sep=",")

data['Estrato'] = data.Estrato.str.split(' ',expand=True)[1]

data['Parking'] = np.where( data.Parking == 'Más de 10', 11, data.Parking)

data_casa = data.query("property_types=='Casa'")\
        [['location_neighbourhoods', 'location_cities',
        'Price', 'Built meters', 'Private square meters',
        'Estrato', 'Rooms', 'Bathrooms', 'Parking', 'Antiquity']]

data_apartamentos = data.query("property_types=='Apartamento'")\
        [['location_neighbourhoods', 'location_cities',
        'Price', 'Built meters', 'Private square meters',
        'Estrato', 'Rooms', 'Bathrooms', 'Parking', 'Antiquity']]

variables_numericas = ['Price',  'Built meters', 'Private square meters',
                        'Rooms', 'Bathrooms', 'Parking', 'Estrato']

variables_categoricas = ['location_neighbourhoods', 'location_cities',  'Antiquity']


data[data=='Sin especificar'].count()

def reemplazar(valor):
        if (valor == 'Sin especificar')|(valor == 'sin especificar'):
                return None
        return valor

data_casa[variables_numericas] = data_casa[variables_numericas].applymap(reemplazar)



#----------------------------------------------
# Gráficos de distribución de densidad: variables numéricas
#----------------------------------------------
# Correlograma
# def corr_graph(df: pd.DataFrame):
    
#     ''' Esta función cálcula la matriz de correlación de las variables en la información ingresada '''
    
#     corr_matrix = df.corr(method='pearson')
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
#     sns.heatmap(
#     corr_matrix,
#     annot     = True,
#     cbar      = False,
#     annot_kws = {"size": 10},
#     vmin      = -1,
#     vmax      = 1,
#     center    = 0,
#     cmap      = sns.diverging_palette(20, 220, n=200),
#     square    = True,
#     ax        = ax
#     )
    
#     ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation = 45,
#     horizontalalignment = 'right')
    
#     ax.tick_params(labelsize = 12)


data.location_neighbourhoods.value_counts()
data.Antiquity.value_counts().plot(kind='bar')
data.location_cities.value_counts().plot(kind='bar')
pd.to_numeric(data.Price).plot(kind='hist', bins=20)
pd.to_numeric(data['Built meters']).value_counts().plot(kind='hist', bins=20)
pd.to_numeric(data['Private square meters']).plot(kind='hist', bins=5)
pd.to_numeric(data.Estrato).plot(kind='hist', bins=20)
pd.to_numeric(data.Rooms, errors='coerce').plot(kind='hist', bins=20)
pd.to_numeric(data.Bathrooms, errors='coerce').plot(kind='hist', bins=20)
pd.to_numeric(data.Parking, errors='coerce').plot(kind='hist', bins=20)


#----------------------------------------------
# Procesamiento de variables explicativas
#----------------------------------------------

data = data_casa

variables_numericas = ['Built meters', 'Private square meters',
                        'Rooms', 'Bathrooms', 'Parking', 'Estrato']

datos_modelo = data.drop(['Price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
                                        datos_modelo,
                                        data['Price'],
                                        train_size   = 1832,
                                        random_state = 111111,
                                        shuffle      = True
                                )


#Pronostico especifico
casa_21=pd.read_excel('C:/Users/luispe/OneDrive - Corporación Interactuar/Aleatorio/Documentos Personales/Proyecto Inmobiliario/Pronostico_inmuebles especificos.xlsx')


X_test = pd.concat([X_test,casa_21])

#----------------------------------------------
# Transformaciones para las variables numéricas
#----------------------------------------------
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True) )])

#----------------------------------------------
# Transformaciones para las variables categóricas
#----------------------------------------------
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

#----------------------------------------------
# Preprocesador
#----------------------------------------------
preprocessor = ColumnTransformer(
                transformers=[
                        ('numeric', numeric_transformer, variables_numericas),
                        ('cat', categorical_transformer, variables_categoricas)],
                remainder='passthrough'
                )

# Generar datos preprocesados
X_train_preprocess = preprocessor.fit_transform(X_train)


# Crear dataframe de datos preprocesados
encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(variables_categoricas)
labels = np.concatenate([variables_numericas, encoded_category])

X_train = pd.DataFrame(preprocessor.transform(X_train), columns=labels)
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=labels)


# reg = LinearRegression().fit(X_train, y_train)
# reg.score(X_train, y_train)
# coeficientes =reg.coef_
# reg.intercept_


# predicciones = reg.predict(X_test)
# mean_absolute_percentage_error(y_test, predicciones)


# reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
# models,predictions = reg.fit(X_train, X_test, y_train, y_test)


regr = RandomForestRegressor(random_state=0)
regr.fit(X_train, y_train)


predicciones = regr.predict(X_test)
mean_absolute_percentage_error(y_test, predicciones[:-1])



