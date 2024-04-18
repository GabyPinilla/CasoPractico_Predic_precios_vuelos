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

## Resultados principales

## Conclusión

## Referencias
