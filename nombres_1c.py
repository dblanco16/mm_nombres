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
generos_por_anio.reset_index(level=['genero_pred'], inplace=True)

generos_por_anio_desglose = pd.DataFrame() 
generos_por_anio_desglose['femenino'] = generos_por_anio['cantidad'][generos_por_anio['genero_pred'] == 'f']
generos_por_anio_desglose['masculino'] = generos_por_anio['cantidad'][generos_por_anio['genero_pred'] == 'm']
 
# guarda sumarizado        
archivo         = 'predict_por_anio_genero.csv'
path_archivo    = directorio_salida+archivo
generos_por_anio_desglose.to_csv(path_archivo, encoding='utf-8', index=True)

# grafica
generos_por_anio_desglose.plot( \
                                   x="anio", \
                                   y=["femenino", "masculino"], \
                                   kind="bar", \
                                   width=0.8, \
                                   title="Cantidad de generos por año", \
                                   fontsize = 8 \
                                )
