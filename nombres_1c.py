# -*- coding: utf-8 -*-

# Parte 1
# C) Usá el clasificador que construiste en el punto anterior y graficá la cantidad estimada de
# nombres de varón y nombres de mujer para cada año.

import pandas as pd

# datos
directorio          = 'F:/modernizacion/'
directorio_salida   = 'F:/modernizacion/salidas/'

archivo             = 'original_predict.csv'
path_archivo        = directorio_salida+archivo
data_nombres_predict = pd.read_csv(path_archivo)

# prepara datos
generos_por_anio = data_nombres_predict.groupby(['anio','genero_pred'])['cantidad'].agg('sum').to_frame()
generos_por_anio.reset_index(level=generos_por_anio.index.names, inplace=True)

registros    = []
generos_por_anio_desglose = pd.DataFrame()
anios = generos_por_anio['anio'].unique()
for anio in anios:
    valores_anio = generos_por_anio[generos_por_anio['anio'] == anio].iloc[:,1:3]
    
    cantidad_f  = 0
    cantidad_m  = 0
    registro    = []
    for index, generos in valores_anio.iterrows():
        if generos['genero_pred'] == 'f':
            cantidad_f = generos['cantidad']
        if generos['genero_pred'] == 'm':
            cantidad_m = generos['cantidad']
    
    registro = [anio, cantidad_f, cantidad_m]
    registros.append(registro)

generos_por_anio_desglose = pd.DataFrame(registros) 
generos_por_anio_desglose.columns = ['anio', 'femenino', 'masculino']
 
# guarda sumarizado        
archivo         = 'predict_por_anio_genero.csv'
path_archivo    = directorio_salida+archivo
generos_por_anio_desglose.to_csv(path_archivo, encoding='utf-8', index=False)

# grafica
generos_por_anio_desglose.plot( \
                                   x="anio", \
                                   y=["femenino", "masculino"], \
                                   kind="bar", \
                                   width=0.8, \
                                   title="Cantidad de generos por año", \
                                   fontsize = 8 \
                                )
