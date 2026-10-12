# Predicción del precio de casas

Proyecto de machine learning supervisado que genera un conjunto de datos inmobiliarios sintéticos, entrena una regresión lineal, evalúa su desempeño y permite usar el modelo desde una aplicación de consola.

> **Alcance:** es un ejercicio educativo con datos simulados. El modelo no está entrenado con ventas reales y sus estimaciones no son tasaciones ni recomendaciones financieras.

## Contenido del proyecto

```text
Proyecto_Casas_ML/
├── notebook_entrenamiento.ipynb  # Generación de datos, entrenamiento, evaluación y exportación
├── datos.pkl                     # Dataset sintético de 500 filas
├── modelo_casas.pkl              # Regresión lineal serializada con Joblib
├── usar_modelo.py                # Aplicación interactiva para realizar predicciones
└── README.md                     # Documentación
```

## Cómo se creó el dataset

El notebook fija la semilla aleatoria en `42` y genera 500 registros. Las variables de entrada son:

| Campo | Descripción | Generación |
| --- | --- | --- |
| `Metros_Cuadrados` | Área de la vivienda | Distribución normal con media 120 y desviación 40; limitada al intervalo de 40 a 400 m² |
| `Habitaciones` | Número de habitaciones | Entero aleatorio entre 1 y 5 |
| `Antiguedad` | Antigüedad en años | Entero aleatorio entre 0 y 49 |
| `Precio` | Variable objetivo en USD | Fórmula sintética descrita abajo |

El precio se genera con esta regla y un término de ruido aleatorio:

```text
Precio = 1500 × Metros_Cuadrados
       + 10000 × Habitaciones
       - 800 × Antiguedad
       + 50000
       + ruido
```

El ruido sigue una distribución normal con media 0 y desviación 15,000. Por eso, el modelo aprende una relación creada artificialmente; el resultado no representa el mercado inmobiliario real.

## Entrenamiento y evaluación

El notebook separa las características (`Metros_Cuadrados`, `Habitaciones`, `Antiguedad`) de la variable objetivo (`Precio`). Divide los datos en 80 % para entrenamiento y 20 % para prueba con `random_state=42`, entrena `LinearRegression` y calcula estas métricas sobre el conjunto de prueba:

| Métrica | Resultado del notebook | Interpretación |
| --- | ---: | --- |
| MSE | 180,449,308.92 USD² | Promedio de los errores de predicción al cuadrado; queda expresado en dólares al cuadrado. |
| R² | 0.9511 | En esta partición sintética, el modelo explica aproximadamente el 95.11 % de la variación observada en los precios de prueba. |

R² **no significa 95.11 % de precisión** ni garantiza que las predicciones tengan ese nivel de acierto en viviendas reales. Las métricas reflejan únicamente este dataset sintético y esta división de prueba.

## Requisitos

- Python 3
- NumPy
- pandas
- scikit-learn
- Joblib

Para ejecutar el predictor local en Windows, instala sus dependencias:

```cmd
py -m pip install joblib scikit-learn
```

Para volver a ejecutar el notebook de entrenamiento, usa Google Colab o un entorno Python con NumPy, pandas, scikit-learn y Joblib. El notebook monta Google Drive y guarda los artefactos en una ruta de Colab; si lo ejecutas fuera de Colab, adapta esa parte a tu entorno.

## Uso del predictor local

Desde la carpeta principal `Modelos De ML Y DL`:

```cmd
py Proyecto_Casas_ML\usar_modelo.py
```

O entra primero en la carpeta del proyecto:

```cmd
cd Proyecto_Casas_ML
py usar_modelo.py
```

El script localiza `modelo_casas.pkl` en la misma carpeta del propio script. Solicita tres datos, uno a la vez, en este orden:

1. Metros cuadrados.
2. Número de habitaciones.
3. Antigüedad en años.

Por ejemplo, para una vivienda de 120 m², 4 habitaciones y 7 años, ingresa `120`, `4` y `7`. Después de cada predicción, responde `S` para calcular otra o `N` para terminar. El modelo se carga una sola vez mientras el programa está abierto.

## Artefacto y compatibilidad

`modelo_casas.pkl` se serializó con Joblib. El notebook reporta la versión de scikit-learn 1.6.1 al guardar el estimador; el predictor se probó con scikit-learn 1.9.1 y mostró el aviso de diferencia de versiones. La predicción de prueba funcionó, pero scikit-learn no garantiza compatibilidad entre versiones para modelos serializados. Para reproducir el entorno de entrenamiento, instala scikit-learn 1.6.1.

Carga archivos `.pkl` únicamente si confías en su origen, ya que la deserialización puede ejecutar código.

## Autor

Proyecto de portafolio personal de Diego Muria.
