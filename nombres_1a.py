# -*- coding: utf-8 -*-

# Parte 1
# A) ¿Cuántos nombres distintos hay en todo el dataset, considerando cada nombre por
# separado? Es decir, un nombre compuesto como María Inés cuenta una vez para María
# y otra para Inés.

import pandas as pd
#import numpy as np
import time

# datos
directorio          = 'F:/modernizacion/'
directorio_salida   = 'F:/modernizacion/salidas/'

data_nombres = pd.read_csv(path_archivo)
# datos fin

# cantidad nombres distintos
columna_nombre      = data_nombres['nombre'].astype('str')
#columna_nombre      = columna_nombre[0:40000] # sacar
columna_nombre = columna_nombre.drop_duplicates()
del(data_nombres)

# limpieza
nombres_excluidos   = columna_nombre.str.contains(r"\d+")
columna_nombre      = columna_nombre[nombres_excluidos == False]

del(nombres_excluidos)
columna_nombre = columna_nombre.reset_index(drop=True)

columna_nombre      = columna_nombre.str.strip()
columna_nombre      = columna_nombre.str.lower()

nombres = columna_nombre.str.split(expand=True)

# guarda nombres separados
archivo         = 'nombres_split.csv'
path_archivo    = directorio_salida+archivo

#nombres = nombres.reset_index(drop=True)
nombres_separados = pd.concat([columna_nombre, nombres], axis= 1)
nombres_separados.to_csv(path_archivo, encoding='utf-8', index=False)

del(columna_nombre)
del(nombres_separados)

# agrupa
nombres_distintos   = pd.DataFrame(columns=['nombre'])

columnas = []
cantidad_columnas = len(nombres.columns)
for i in range(0,cantidad_columnas):
    print("AGRUPA i=> "+ str(i))
    columnas.append(nombres[i][nombres[i].notnull()])
nombres_distintos = pd.concat(columnas, ignore_index=True).to_frame()
nombres_distintos.columns = ['nombre']
nombres_distintos = nombres_distintos.drop_duplicates()

del(nombres)

# limpieza
starting_point = time.time()

# limpia caracteres
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('\"', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('\/', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace(',', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('.', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('_', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('\´', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('\`', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('(', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace(')', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('-', "")
nombres_distintos['nombre']   = nombres_distintos['nombre'].str.replace('\'', "")

# caracter solos
nombres_excluidos = (nombres_distintos['nombre'].str.len() < 2)
nombres_distintos = nombres_distintos[~nombres_excluidos]

# conectores
excluir = ['del', 'de', 'los', 'lo', 'la', 'las']
nombres_excluidos = nombres_distintos['nombre'].isin(excluir)
nombres_distintos = nombres_distintos[~nombres_excluidos]

nombres_distintos = nombres_distintos.drop_duplicates()

del(nombres_excluidos)

# ordena
print("ordern => ")
starting_point = time.time()

nombres_distintos = nombres_distintos.sort_values(by='nombre', ascending=True)            

cantidad_distintos = len(nombres_distintos)
print("Nombres distintos => "+str(cantidad_distintos))

archivo         = 'nombres_distintos.csv'
path_archivo    = directorio_salida+archivo
nombres_distintos.to_csv(path_archivo, encoding='utf-8', index=False)
