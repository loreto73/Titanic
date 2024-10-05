#Importamos librerías necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creamos el dataframe
Titanic = pd.read_csv('/home/luis-loreto/Documentos/Python/Titanic/Tablas/train.csv')

#Hay dos variables problemáticas: Age y Ticket. La primera requiere algún método de imputación para poder usarse y la segunda reuquiere limpieza

#Creamos una función para la limpieza de teciket y sobreescribimos el dataframe.
def limpieza(texto):
    # Eliminar el contenido antes del primer espacio
    parte_despues_espacio = texto.split(' ', 1)[1] if ' ' in texto else texto
    # Retornar el primer carácter de esa parte
    return parte_despues_espacio[0] if parte_despues_espacio else ''

Titanic['Ticket'] = Titanic['Ticket'].apply(limpieza)

#Ahora usamos el método hot deck para la imputación
def hot_deck(Titanic):
    for column in Titanic.columns:
        # Encontrar índices donde hay valores faltantes
        missing_indices = Titanic[Titanic[column].isnull()].index
        
        for idx in missing_indices:
            # Seleccionar filas no faltantes
            non_missing_values = Titanic[column].dropna()
            # Elegir un valor aleatorio de las filas no faltantes
            random_value = non_missing_values.sample(n=1).values[0]
            # Imputar el valor aleatorio en el índice correspondiente
            Titanic.at[idx, column] = random_value
            
    return Titanic

Titanic = hot_deck(Titanic)

print(Titanic)