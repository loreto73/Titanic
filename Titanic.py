import pandas as pd
df = pd.read_csv('/home/luis-loreto/Documentos/Python/Titanic/Tablas/train.csv')

df.count()

resultado = df.groupby(["Survived", "Pclass"])["Survived"].count().reset_index(name="Cantidad")