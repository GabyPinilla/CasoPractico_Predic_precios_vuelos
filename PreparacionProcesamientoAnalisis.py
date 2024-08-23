# Importando librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import joblib

# PREPARACION DE LOS DATOS
# Cargando datos
df_economy = pd.read_excel('C:\\Users\\gabri\\Desktop\\proyectos\\economy.xlsx')
df_business = pd.read_excel('C:\\Users\\gabri\\Desktop\\proyectos\\business.xlsx')

# Visualizando datasets
df_economy.head()
df_business.head()

# Creando columna class en ambos df
df_business['class'] = 'business'
df_economy['class'] = 'economy'

# Concatenando los df
df = pd.concat([df_business, df_economy], axis=0)
df.head()

# Obteniendo los nombres de las variables con sus tipos de datos
df.info()

# Transformando el tipo de dato de price
df['price'] = pd.to_numeric(df['price'], errors='coerce')
# Verificando existencia de datos nulos
df.isnull().sum()
# 108 datos nulos de 300261

# Eliminando filas con datos nulos
df.dropna(inplace=True)

# Transformacion de time_taken para obtener el tiempo de vuelo en minutos
df[['hours', 'minutes']] = df['time_taken'].str.split('h ', expand=True)
df['minutes'] = pd.to_numeric(df['minutes'].str.replace('m', ''))
df['hours'] = pd.to_numeric(df['hours'])
# Transformando a minutos
df['time_taken'] = df['hours'] * 60 + df['minutes']
# Eliminando hours y minutes
df.drop(['hours', 'minutes'], axis=1, inplace=True)
df.head()

# Visualizando datos unicos de stop
df['stop'].unique()

# Extrayendo numero de escalas
df['stop'] = df['stop'].str.split('-').str[0]
df['stop'] = pd.to_numeric(np.where(df['stop'] == 'non', 0,
                       np.where(df['stop'] == '2+', 2, 1)))
# Verificando la tranformacion
df['stop'].unique()
# La transformacion fue realizada con exito

# Visualizando variables categoriccas del df
cat_cols = df.select_dtypes(['object']).columns
for i in cat_cols:
  print(df[i].value_counts(normalize=True))
  print('-----------------------------')

# Eliminando arr_time y dep_time
df.drop(['arr_time', 'dep_time'], axis=1, inplace=True)

# Creando weekday para obtener los dias de la semana
df['weekday'] = df['date'].dt.weekday
df['weekday'].replace({0: 'Monday',
                       1: 'Tuesday',
                       2: 'Wednesday',
                       3: 'Thursday',
                       4: 'Friday',
                       5: 'Saturday',
                       6: 'Sunday'}, inplace=True)

# Eliminando variables
df.drop(['ch_code', 'num_code', 'date'], axis=1, inplace=True)

# Eliminando datos nulos
df.dropna(inplace=True)

# Visualizando distribucion de los datos
df.describe()

# Visualizando distribucion de price y time_taken mediante boxplot
numeric_cols = ['price', 'time_taken']
for n in numeric_cols:
  sns.boxplot(data=df, x=n)
  plt.show()

# Visualizando los precios con respecto a la clase, escala y dia de la semana
cols = ['class', 'stop', 'weekday']
for i in cols:
  sns.boxplot(data=df, x=i, y='price')
  plt.title(f'{i} y price')
  plt.show()

# Copia de df
df_original = df.copy()

# LIMPIEZA DE DATOS
# Tratando datos atipicos con z score
# Funciones z score
def detect_outliers_zscore(data, threshold=2):
    # Calcular el Z-score para cada punto de datos
    z_scores = (data - np.mean(data)) / np.std(data)

    # Encontrar valores atipicos basados en el umbral
    outliers = np.abs(z_scores) > threshold

    return outliers

def plot_outliers(data, outliers):
    # Crear un gráfico de dispersion para los valores normales
    plt.figure(figsize=(10, 6))
    plt.scatter(data[~outliers], [1] * len(data[~outliers]), label='Valores Normales', color='blue', s=50)

    # Crear un gráfico de dispersion para los valores atípicos
    plt.scatter(data[outliers], [1] * len(data[outliers]), label='Valores Atípicos', color='red', marker='x', s=100)

    plt.title('Detección de Outliers con Z-score (tresh=2)')
    plt.xlabel('Datos')
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    plt.show()

# Aplicando z score al df
for n in numeric_cols:
  print(f'{n}')
  outliers = detect_outliers_zscore(df[n], threshold=2)
  plot_outliers(df[n], outliers)
  df = df[~outliers]
# Reseteando los indices
df.reset_index(drop=True, inplace=True)

# Cantidad de datos atipicos eliminados con z score
df_original.shape[0] - df.shape[0]

# ANALISIS
# Correlacion de las variables numericas
df_corr = df[['price','time_taken','stop']]
corr = df_corr.corr()
mask= np.zeros_like(corr) # Mascara
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,
            vmax=1, vmin=-1,
            annot=True, annot_kws={'fontsize':7},
            mask=mask, # Aplicando mascara
            cmap=sns.diverging_palette(20,220,as_cmap=True))

# Definiendo Economy = 0, Business=1
df['class'] = np.where(df['class'] == 'economy', 0, 1)

# Transformando datos a binario
cat_cols = df.select_dtypes(['object']).columns
data_cat = pd.get_dummies(df[cat_cols], drop_first=True)
model_df = pd.concat([data_cat, df[['time_taken', 'price', 'class', 'stop']]], axis=1)

# Calculando correlacion con variables binarias
corr = model_df.corr(method='pearson')['price']
df_corr = pd.DataFrame(corr)
df_corr.sort_values('price',ascending=False)

# MODELANDO
# Aplicando modelos
model_df.dropna(inplace=True)
# Caracteristicas
X = model_df.drop('price', axis=1)
# Target
y = model_df['price']
# Dividiendo el set de datos en conjunto de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos a aplicar
models = [
    {
        'name': 'LinearRegression',
        'model': LinearRegression(),
        'params': {}
    },
    {
        'name': 'Ridge',
        'model': Ridge(),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}
    },
    {
        'name': 'Lasso',
        'model': Lasso(),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}
    },
    {
        'name': 'ElasticNet',
        'model': ElasticNet(),
        'params': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
    }
]

# Escogiendo los mejores hiperparametros para cada modelo
best_models = {}
for model_info in models:
    model = model_info['model']
    model_name = model_info['name']
    param_grid = model_info['params']

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_models[model_name] = grid_search.best_estimator_

    print(f"Mejores hiperparametros para {model_name}: {grid_search.best_params_}")

# Evaluando los modelos y guardandolos en un archivo pickle
for model_name, model in best_models.items():
  joblib.dump(model, f"{model_name}_model.pkl")

model_names = ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]
model_files = ["LinearRegression_model.pkl", "Ridge_model.pkl", "Lasso_model.pkl", "ElasticNet_model.pkl"]

results = []

for model_name, model_file in zip(model_names, model_files):
  model = joblib.load(model_file)
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  ape = np.abs((y_test - y_pred) / y_test)
  mape = np.mean(ape) * 100

  results.append(pd.DataFrame({"Model": [model_name], "R^2": [r2], "MAE": [mae], "MAPE": [mape]}))

results = pd.concat(results).reset_index(drop=True)
# Imprimiendo resultados
print(results)

# Graficando los valores reales y predicciones
for model_name, model_file in zip(model_names, model_files):
  model = joblib.load(model_file)
  y_pred = model.predict(X_test)
  sns.regplot(x=y_test, y=y_pred, ci=None, line_kws=dict(color="g"))
  plt.xlabel('Valores reales')
  plt.ylabel('Predicciones')
  plt.title(f'Predicciones vs Valores reales: {model_name}')
  plt.savefig(f'{model}.png') # Descarga las graficas de los modelos
  plt.show()
# Los resultados muestran que Lasso es el modelo con mejores metricas, el cual es usado para calcular los coeficientes

# Coeficientes de las variables
# Caracteristicas
X = model_df.drop('price', axis=1)
# Target
y = model_df['price']
# Dividiendo el set de datos en conjunto de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Instanciando y entrenando el modelo
lasso = Lasso(alpha= 0.01)
lasso.fit(X_train, y_train)
# Predicciones
y_pred = lasso.predict(X_test)

# Coeficientes de Lasso
coefficients = pd.DataFrame(lasso.coef_, model_df.drop('price', axis = 1).columns)
coefficients.columns = ['Coefficients']
coefficients.sort_values(by='Coefficients', ascending=False)
