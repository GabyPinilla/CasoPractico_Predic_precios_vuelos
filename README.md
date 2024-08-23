# Caso práctico de predicción de precios de vuelos

## Introducción
A partir de datos de vuelos, se quiere predecir el valor del boleto de un avión usando los siguientes algoritmos de aprendizaje supervisado:
- Regresión lineal
- Lasso
- Ridge
- ElasticNet

Mediante el análisis, limpieza y posterior aplicación de los modelos a los datos, se evaluará el rendimiento de cada uno con las métricas de evaluación para definir el más eficiente y generar una conclusión a partir de los resultados obtenidos.

## Fuente de datos
Los datos de los vuelos están divididos en dos datasets: *economy* y *business*. Ambos poseen la misma cantidad de campos sobre los vuelos durante el mes de marzo de 2022, como la fecha, aerolínea, código, número de vuelo, hora de salida, horaq de llegada, duración, escalas, aeropuerto de destino y el precio del boleto.

Estos datasets están disponibles en el siguiente link de Kaggle y son de dominio público:
[https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

## Procesamiento de datos
### Modificación de datos
1. Creación de una nueva variable llamada *class* en cada dataset para diferenciar los datos al momento de concatenar los datasets.
2. Unión de los datasets para agilizar el análisis.
3. Cambio en los tipos de datos del precio a numérico.
4. Transformación de la variable de la duración del vuelo para obtener el tiempo de vuelo en minutos en un formato numérico.
5. Extracción de los números de paradas de la variable escala debido a la presencia de datos incoherentes.
6. Creación de una nueva variable llamada días de la semana a partir de la fecha para trabajar con los días de los vuelos.

### Eliminación de datos
1. Eliminación de datos nulos al modificar variables debido a que representaban un porcentaje insignificante del conjunto de datos.
2. Eliminación de hora de llegada y hora de salida de los vuelos para trabajar solo con la duración total del veulo.
3. Eliminación de la fecha y variables relacionadas con los códigos que no aportan al análisis.

## Resumen del análisis
Se quiere analizar el comportamiento de los precios dependiendo de la duración del vuelo, clase, cantidad de escalas y día de la semana en la que se realizó el vuelo. Mediante gráficos boxplot, se observó la distribución de los precios en cada variable mencionada.

Para poder aplicar los modelos regresión lineal, rigde, lasso y elasticnet, las variables se tranformaron a binarias y se compararon sus correlaciones con respecto al precio. Luego, se evaluaron los modelos con $R^{2}$, MAE y MAPE para comparar las métricas y definir el modelo más eficiente para resolver el problema. Finalmente, se calcularon los coeficientes de las variables del modelo elegido.

### Resultados principales
Los princioales resultados obtenidos a partir del análisis fueron:
- En promedio, los tiempos de vuelo del dataset duran alrededor de 733 minutos, donde la hora de salida es al mediodía y la llegada a las 15 horas.
- La clase *business* tiende a tener un mayor precio promedio que la *economy*.
- Los vuelos que presentan 1 o 2 escalas son más caros que los vuelos directos.
- No existe una diferencia significativa entre los precios de los vuelos entre los días de la semana y los fines de semana.

#### Métricas de evaluación
|Modelo| $R^2$ |MAE|MAPE|
|:---:|:---:|:---:|:---:|
|Lineal|  0.920154|  4156.253284|  43.253665|
|Ridge|  0.920154|  4156.257065|  43.253623|
|Lasso|  0.920154|  4156.237780|  43.253089|
|ElasticNet|  0.920019|  4177.404655|  43.360865|

Los resultados obtenidos luego de aplicar los modelos fueron:
- La clase posee una mayor correlación con el precio de los vuelos.
- Los modelos regresión lineal, ridge y lasso poseen un rendimiento muy similar, con una ligera variación en sus métricas. A pesar de estas diferencias no muy significativas, se optó por definir lasso como el modelo más eficiente.
- Las gráficas muestran que existe un factor que está afectando las predicciones del modelo, por lo que al obtener el coeficiente de las variables se llegó a la conclusión de que la clase y la cantidad de escalas son las que más influyen en los precios de los voletos.

## Conclusión
De lo anterior, se puede concluir que los modelos regresión lineal, ridge y lasso funcionan de forma eficiente para este problema, sin embargo, el modelo con mejores métricas fue lasso. A partir de esto, resultó que las variables que poseen una fuerte relación con el precio del boleto del avión fueron la clase y la cantidad de paradas. La clase economy tiende a tener precios más económicos que la business. Por su parte, las escalas oscilan entre 0 y 2, donde un vuelo con 1 o 2 paradas tiende a ser más caro que sin paradas.

Con esto, podrían tomarse medidas como:
- disminuir la cantidad de escalas en vuelos de corta distancia
- ofrecer tarifas de acuerdo con la cantidad de turistas, como a familias, estudiantes, parejas.

## Referencias
