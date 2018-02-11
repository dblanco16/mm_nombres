# -*- coding: utf-8 -*-

# Parte 2
# En esta parte te pedimos que realices un pequeño análisis exploratorio del dataset y que lo
# presentes en forma de reporte breve en un archivo PDF. 
# Tanto la diagramación del informe como el contenido queda a tu criterio.
# Solo a modo disparador incluímos algunas preguntas:
# ¿Qué preguntas interesantes se pueden responder con los datos? 
# ¿Qué cosas en el dataset son anómalas y llevan a pensar en posibles errores? 
# ¿Con qué otra fuente de datos sería interesante integrar este dataset?

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns

# FUNCIONES #
def limpia_caracteres(lista_caracteres, dataframe, campo):
    for c in lista_caracteres:
        dataframe[campo]  = dataframe[campo].str.replace(c, "")
        
def grafica_periodo(data_decada, inicio, fin, cantidad_top):
    columnas = data_decada.columns
    total_registors = data_decada.shape[0]
    for campo in columnas:
        # cantidad nulos
       cant_nulos = data_decada.loc[:, campo].isnull().sum()
       
       if cant_nulos == total_registors:
           data_decada = data_decada.drop(campo, 1)

    fig = plt.figure(figsize=(12,12))
    r = sns.heatmap(data_decada, cmap='BuPu')
    r.set_title("Top "+str(cantidad_top)+" nombres desde "+str(inicio)+" a "+str(fin))
    r.set(xlabel ='Nombres', ylabel='Años')
    plt.xticks(rotation=45)
        
# FUNCIONES #
        
# datos
directorio          = 'F:/modernizacion/local/'
directorio_salida   = 'F:/modernizacion/local/salidas/'

archivo             = 'historico-nombres.csv'
path_archivo        = directorio+archivo
data_nombres        = pd.read_csv(path_archivo)

archivo             = 'seleccionados.csv'
path_archivo        = directorio+archivo
data_seleccionados  = pd.read_csv(path_archivo, sep=";")

# control datos
print ("Registros => "+str(data_nombres.shape[0]))

# nulos
frec_nulos = data_nombres.isnull().sum()
print ("Valores nulos => ")
frec_nulos

# digitos
nombres_con_digitos = data_nombres.nombre.str.contains("\d+")
frec_digitos = data_nombres[nombres_con_digitos == True].shape[0]
del nombres_con_digitos
print ("Nombres con digitos => ")
frec_digitos

# caracteres "erroneos"
# caracteres = [ '\"', '\/',  ',', '.', '\´', '\`', '(', ')', '-', '\'']
nombres_con_caracteres = data_nombres.nombre.str.contains("[\"]+|[\/]+|[,]+|[.]+|[\´]+|[\´]+|[\()]+|[\)]+|[\-]+|[\']+")
frec_caracteres = data_nombres[nombres_con_caracteres == True].shape[0]
del nombres_con_caracteres
print ("Nombres con caracteres erroneos => ")
frec_caracteres

# duplicados (nombre, anio)
registros_duplicados =  data_nombres.duplicated(subset=['nombre','anio'], keep=False).astype(int).astype(str)
frec_duplicados = data_nombres[registros_duplicados == True].shape[0]
del registros_duplicados

print ("Registros duplicados => ")
frec_duplicados

# longitud
registros_cortos = (data_nombres['nombre'].str.len() < 2)
frec_long = data_nombres[registros_cortos == True].shape[0]
del registros_cortos
print ("Registros cortos => ")
frec_long

# exploratorio originales limpios
# elimina digitos
nombres = data_nombres.copy()

print ("Nombres total => "+str(nombres.shape[0]))

nombres_con_digitos = nombres.nombre.str.contains("\d+")
nombres             = nombres[nombres_con_digitos == False]
del nombres_con_digitos

# reemplaza caracteres erroneos
caracteres = [ '\"', '\/',  ',', '.', '\´', '\`', '(', ')', '-', '\'']
limpia_caracteres(caracteres, nombres, 'nombre')

# convierte a minusculas y sin espacios
nombres['nombre']      = nombres['nombre'].str.strip()
nombres['nombre']      = nombres['nombre'].str.lower()

# elimina nulos
nombres         = nombres[~nombres.nombre.isnull()]

# longitud casi nula
registros_cortos_limpios = (nombres['nombre'].str.len() < 2)
nombres             = nombres[registros_cortos_limpios == False]
del registros_cortos_limpios

# duplicados (nombre, anio)
registros_duplicados_limpios =  nombres.duplicated(subset=['nombre','anio'], keep=False).astype(int).astype(str)
frec_duplicados_limpios = nombres[registros_duplicados_limpios == True].shape[0]
del registros_duplicados_limpios

print ("Registros duplicados limpios => ")
frec_duplicados_limpios

print ("Nombres total modificados => "+str(nombres.shape[0]))

# tabla de frecuencia de nombres
frec_nombres = pd.value_counts(nombres['nombre'])
maxima_repeticion_nombre = frec_nombres[frec_nombres.idxmax()]
minima_repeticion_nombre = frec_nombres[frec_nombres.idxmin()]

print ("Frecuencia de nombres => "+str(frec_nombres.shape[0]))
print ("Frecuencia de nombres maxima => "+frec_nombres.idxmax()+" "+str(maxima_repeticion_nombre))
print ("Frecuencia de nombres minima=> "+frec_nombres.idxmin()+" "+str(minima_repeticion_nombre))

# tabla de frecuencia de anios
frec_anios = pd.value_counts(nombres['anio'])
maxima_repeticion_anio = frec_anios[frec_anios.idxmax()]
minima_repeticion_anio = frec_anios[frec_anios.idxmin()]

print ("Frecuencia de anios => "+str(frec_anios.shape[0]))
print ("Frecuencia de anios maxima => "+str(frec_anios.idxmax())+" "+str(maxima_repeticion_anio))
print ("Frecuencia de anios minima=> "+str(frec_anios.idxmin())+" "+str(minima_repeticion_anio))

nombres.anio.min()
nombres.anio.max()
frec_anios[2011] # pico bajo atipico

# exploratorio originales limpios fin
 
# cantidad nombres por anio
anios_cantidad = nombres.groupby(['anio'])['cantidad'].agg('sum').to_frame()
anios_cantidad.head()

print ("Frecuencia de nombres maxima => "+str(anios_cantidad.cantidad.idxmax())+" "+str(anios_cantidad.cantidad[anios_cantidad.cantidad.idxmax()]))
print ("Frecuencia de anios minima=> "+str(anios_cantidad.cantidad.idxmin())+" "+str(anios_cantidad.cantidad[anios_cantidad.cantidad.idxmin()]))

# grafico barras
fig, ax = plt.subplots()
anios_cantidad.plot(kind='bar', width=0.6, ax=ax);
ax.grid()

# grafico dos ejes
anios_cantidad.reset_index(inplace=True)
anios_cantidad.head()

ax = anios_cantidad.plot(x='anio', y='cantidad' ,figsize=(10,5), grid=True)
ax.set_xlim((anios_cantidad['anio'].min(), anios_cantidad['anio'].max()))
ax.set_xlabel("Años")
ax.set_ylabel("Cantidad de nombres")
ax.xaxis.set_ticks(np.arange(anios_cantidad['anio'].min(), anios_cantidad['anio'].max(), 2))
for tick in ax.get_xticklabels():
        tick.set_rotation(90)

# grafico boxplot cantidad
ax_boxp = anios_cantidad.boxplot(column='cantidad')
ax_boxp.set_ylim((0, anios_cantidad['cantidad'].max()+50000))
ax_boxp.yaxis.set_ticks(np.arange(0, anios_cantidad['cantidad'].max() + 50000, 50000))
        
# descriptivo
anios_cantidad_desc=anios_cantidad.cantidad.describe()
anios_cantidad_desc
# cantidad nombres por anio fin

# prepara datos nombres
nombres_desg = nombres.copy()
nombres_desg['nombres'] = nombres.nombre.str.split()

# saca conectores
excluir = ['del', 'de', 'los', 'lo', 'la', 'las']
set_excluir = set(excluir)

for index, row in nombres_desg.iterrows():
    reg_nombres = set(row['nombres'])
    new_nombres = list(reg_nombres.difference(set_excluir))

    if len(reg_nombres) != len(new_nombres):
        nombres_desg.set_value(index,'nombres',new_nombres)

nombres_desg['nombres_simples'] = nombres_desg['nombres'].str.len()
# saca nulos
registros_validos = (nombres_desg.nombres_simples >0)
nombres_desg = nombres_desg[registros_validos]
del registros_validos

# guarda para reutilizar
archivo         = 'nombres_desg_lista.csv'
path_archivo    = directorio_salida+archivo
nombres_desg.to_csv(path_archivo, encoding='utf-8', index=False)
#nombres_desg        = pd.read_csv(path_archivo)

# cantidad nombres usados
# cantidad nombres por anio y #simples
anios_simples_cantidad = nombres_desg.groupby(['anio','nombres_simples'])['cantidad'].agg('sum').to_frame()
anios_simples_cantidad.head()

anios_simples_cantidad.reset_index(level=['nombres_simples'], inplace=True)
anios_simples_cantidad.head()

anios_simples_cantidad_desglose = pd.DataFrame() 
lista_cantidades = anios_simples_cantidad.nombres_simples.unique()
lista_cantidades.sort()
etiquetas = []
for i in lista_cantidades:
    anios_simples_cantidad_desglose['nombres_'+str(i)] = anios_simples_cantidad['cantidad'][anios_simples_cantidad['nombres_simples'] == i]
    if i>1:
        etiquetas.append(str(i)+' Nombres')
    else:
        etiquetas.append(str(i)+' Nombre')        
anios_simples_cantidad_desglose.head()

# grafico barras
fig, ax = plt.subplots()
colors = ['red', 'yellow', 'green', 'blue', 'fuchsia', 'gold', 'coral', 'lime', 'pink', 'lavender']
anios_simples_cantidad_desglose.plot(kind='bar', width=0.6, ax=ax, stacked=True, color=colors);
ax.yaxis.set_ticks(np.arange(0, 1300000, 100000))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, etiquetas)

# descriptivos
anios_simples_cantidad_desglose.loc[:,['nombres_1','nombres_2','nombres_3']].describe()
anios_simples_cantidad_desglose.loc[:,['nombres_1','nombres_2','nombres_3']].sum()

# porcentajes
for index, row in anios_simples_cantidad_desglose.iterrows():
    total = row.sum(axis=0)
    if total > 0:
        porcentaje = row['nombres_1'] / total
        anios_simples_cantidad_desglose.set_value(index,'nombres_1_porc', porcentaje)
        porcentaje = row['nombres_2'] / total
        anios_simples_cantidad_desglose.set_value(index,'nombres_2_porc', porcentaje)
        porcentaje = row['nombres_3'] / total
        anios_simples_cantidad_desglose.set_value(index,'nombres_3_porc', porcentaje)
    
analisis_anios = anios_simples_cantidad_desglose.loc[:,['nombres_1_porc', 'nombres_2_porc', 'nombres_3_porc']]
    
# boxplot
ax_boxp_nombres = anios_simples_cantidad_desglose.loc[:,['nombres_1','nombres_2','nombres_3']].plot.box( \
                                                           patch_artist=True, \
                                                           color= {'whiskers': 'purple', \
                                                                 'caps': 'fuchsia', \
                                                                 'boxes': 'skyblue', \
                                                                 'medians': 'blue'}) 
maximo_nombres = max(anios_simples_cantidad_desglose['nombres_1'].max(),anios_simples_cantidad_desglose['nombres_2'].max(), anios_simples_cantidad_desglose['nombres_3'].max())
ax_boxp_nombres.set_ylim((0,maximo_nombres))
ax_boxp_nombres.yaxis.set_ticks(np.arange(0, maximo_nombres+100000, 100000))
# cantidad nombres usados fin

# ranking de nombres simples
nombres_simples = pd.DataFrame() 
list_nombres_simples = []
for index, row in nombres_desg.iterrows():
    reg_nombres = row['nombres']

    for reg_nombre in reg_nombres:
        list_nombres_simples.append([reg_nombre, row['cantidad'], row['anio']])
nombres_simples = nombres_simples.append(list_nombres_simples)
del list_nombres_simples

nombres_simples.columns = ['nombre', 'cantidad', 'anio']

anios_nombres_simples = nombres_simples.groupby(['anio','nombre'])['cantidad'].agg('sum').to_frame()

del nombres_simples

# guarda para reutilizar
archivo         = 'nombres_simples.csv'
path_archivo    = directorio_salida+archivo
anios_nombres_simples.to_csv(path_archivo, encoding='utf-8', index=False)
#anios_nombres_simples        = pd.read_csv(path_archivo)

anios_nombres_simples = anios_nombres_simples['cantidad'].groupby(level = 0, group_keys=False)

# top 1
anios_nombres_top1 = anios_nombres_simples.nlargest(n=1, keep="last").to_frame()

for index, row in anios_nombres_top1.iterrows():
    anio = index[0]
    total = anios_cantidad[anios_cantidad.anio == anio].cantidad.values[0]
    anios_nombres_top1.set_value(index,'total', total)
    anios_nombres_top1.set_value(index,'porc_cant', row['cantidad'] / total)

anios_nombres_top1.reset_index(level=['anio', 'nombre'], inplace=True)

# guarda para reutilizar
archivo         = 'nombres_simples_top1.csv'
path_archivo    = directorio_salida+archivo
anios_nombres_top1.to_csv(path_archivo, encoding='utf-8', index=False)
#anios_nombres_top1        = pd.read_csv(path_archivo)

# heat map
# categtoria nombres
anios_nombres_top1["nombre"] = pd.Categorical(anios_nombres_top1["nombre"], anios_nombres_top1.nombre.unique())
# arma matriz
top1_matrix = anios_nombres_top1.pivot("anio", "nombre", "porc_cant")
# ordena por nombre
columnas = sorted(top1_matrix.columns)
top1_matrix = top1_matrix[columnas]

# gaficos por decada
inicio      = 1922
fin         = 1929
data_decada = top1_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 1)

for inicio in range(1930, 2001, 10):
    fin = inicio + 9
    data_decada = top1_matrix.loc[inicio:fin]
    grafica_periodo(data_decada, inicio, fin, 1)

inicio = 2010
fin = 2015
data_decada = top1_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 1)

# top 5
anios_nombres_top5 = anios_nombres_simples.nlargest(n=5, keep="last").to_frame()

for index, row in anios_nombres_top5.iterrows():
    anio = index[0]
    total = anios_cantidad[anios_cantidad.anio == anio].cantidad.values[0]
    anios_nombres_top5.set_value(index,'total', total)
    anios_nombres_top5.set_value(index,'porc_cant', row['cantidad'] / total)

anios_nombres_top5.reset_index(level=['anio', 'nombre'], inplace=True)

# guarda para reutilizar
archivo         = 'nombres_simples_top5.csv'
path_archivo    = directorio_salida+archivo
anios_nombres_top5.to_csv(path_archivo, encoding='utf-8', index=False)
#anios_nombres_top5        = pd.read_csv(path_archivo)

# categtoria nombres
anios_nombres_top5["nombre"] = pd.Categorical(anios_nombres_top5["nombre"], anios_nombres_top5.nombre.unique())
# arma matriz
top5_matrix = anios_nombres_top5.pivot("anio", "nombre", "porc_cant")
# ordena por nombre
columnas = sorted(top5_matrix.columns)
top5_matrix = top5_matrix[columnas]

# gaficos por decada
inicio      = 1922
fin         = 1929
data_decada = top5_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 5)

for inicio in range(1930, 2001, 10):
    fin = inicio + 9
    data_decada = top5_matrix.loc[inicio:fin]
    grafica_periodo(data_decada, inicio, fin, 5)

inicio = 2010
fin = 2015
data_decada = top5_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 5)

# nombres completos
anios_nombres_full = nombres.groupby(['anio','nombre'])['cantidad'].agg('sum').to_frame()
anios_nombres_full = anios_nombres_full['cantidad'].groupby(level = 0, group_keys=False)

# top 1
anios_nombres_full_top1 = anios_nombres_full.nlargest(n=1, keep="last").to_frame()

for index, row in anios_nombres_full_top1.iterrows():
    anio = index[0]
    total = anios_cantidad[anios_cantidad.anio == anio].cantidad.values[0]
    anios_nombres_full_top1.set_value(index,'total', total)
    anios_nombres_full_top1.set_value(index,'porc_cant', row['cantidad'] / total)

anios_nombres_full_top1.reset_index(level=['anio', 'nombre'], inplace=True)

# guarda para reutilizar
archivo         = 'nombres_full_top1.csv'
path_archivo    = directorio_salida+archivo
anios_nombres_full_top1.to_csv(path_archivo, encoding='utf-8', index=False)
#anios_nombres_full_top1        = pd.read_csv(path_archivo)

# categtoria nombres
anios_nombres_full_top1["nombre"] = pd.Categorical(anios_nombres_full_top1["nombre"], anios_nombres_full_top1.nombre.unique())
# arma matriz
top1_full_matrix = anios_nombres_full_top1.pivot("anio", "nombre", "porc_cant")
# ordena por nombre
columnas = sorted(top1_full_matrix.columns)
top1_full_matrix = top1_full_matrix[columnas]

# gaficos por decada
inicio      = 1922
fin         = 1929
data_decada = top1_full_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 1)

for inicio in range(1930, 2001, 10):
    fin = inicio + 9
    data_decada = top1_full_matrix.loc[inicio:fin]
    grafica_periodo(data_decada, inicio, fin, 1)

inicio = 2010
fin = 2015
data_decada = top1_full_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 1)

# top 5
anios_nombres_full_top5 = anios_nombres_full.nlargest(n=5, keep="last").to_frame()

for index, row in anios_nombres_full_top5.iterrows():
    anio = index[0]
    total = anios_cantidad[anios_cantidad.anio == anio].cantidad.values[0]
    anios_nombres_full_top5.set_value(index,'total', total)
    anios_nombres_full_top5.set_value(index,'porc_cant', row['cantidad'] / total)

anios_nombres_full_top5.reset_index(level=['anio', 'nombre'], inplace=True)

# guarda para reutilizar
archivo         = 'nombres_full_top5.csv'
path_archivo    = directorio_salida+archivo
anios_nombres_full_top5.to_csv(path_archivo, encoding='utf-8', index=False)
#anios_nombres_full_top5        = pd.read_csv(path_archivo)

# categtoria nombres
anios_nombres_full_top5["nombre"] = pd.Categorical(anios_nombres_full_top5["nombre"], anios_nombres_full_top5.nombre.unique())
# arma matriz
top5_full_matrix = anios_nombres_full_top5.pivot("anio", "nombre", "porc_cant")
# ordena por nombre
columnas = sorted(top5_full_matrix.columns)
top5_full_matrix = top5_full_matrix[columnas]

# gaficos por decada
inicio      = 1922
fin         = 1929
data_decada = top5_full_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 5)

for inicio in range(1930, 2001, 10):
    fin = inicio + 9
    data_decada = top5_full_matrix.loc[inicio:fin]
    grafica_periodo(data_decada, inicio, fin, 5)

inicio = 2010
fin = 2015
data_decada = top5_full_matrix.loc[inicio:fin]
grafica_periodo(data_decada, inicio, fin, 5)
# ranking de nombres fin

# cruce deportes
nombres_full_top100 = anios_nombres_full.nlargest(n=100, keep="last").to_frame()

# 1978 al 80
deportistas_anio_cantidad = nombres_full_top100.loc[1978:1980]

deportistas = deportistas_anio_cantidad.groupby(['nombre'])['cantidad'].agg('sum').to_frame()

total = anios_cantidad[anios_cantidad.anio == 1978].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1979].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1980].cantidad.values[0]
for index, row in deportistas.iterrows():
    deportistas.set_value(index,'porc_cant', row['cantidad'] / total)

deportistas_top = deportistas.sort_values(['porc_cant'], ascending=[False])
deportistas_top.reset_index(inplace=True)
deportistas_top

# 1986 al 88
deportistas_anio_cantidad = nombres_full_top100.loc[1986:1988]
deportistas = deportistas_anio_cantidad.groupby(['nombre'])['cantidad'].agg('sum').to_frame()

total = anios_cantidad[anios_cantidad.anio == 1986].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1987].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1988].cantidad.values[0]
for index, row in deportistas.iterrows():
    deportistas.set_value(index,'porc_cant', row['cantidad'] / total)

deportistas_top = deportistas.sort_values(['porc_cant'], ascending=[False])
deportistas_top.reset_index(inplace=True)
deportistas_top

# mujeres olimpicas
noemi = nombres_full_top100.loc[1948:1950]
noemi = noemi.groupby(['nombre'])['cantidad'].agg('sum').to_frame()
total = anios_cantidad[anios_cantidad.anio == 1948].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1949].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1950].cantidad.values[0]
for index, row in noemi.iterrows():
    noemi.set_value(index,'porc_cant', row['cantidad'] / total)

noemi_top = noemi.sort_values(['porc_cant'], ascending=[False])
noemi_top.reset_index(inplace=True)

gabriela = nombres_full_top100.loc[1988:1990]
gabriela = gabriela.groupby(['nombre'])['cantidad'].agg('sum').to_frame()
total = anios_cantidad[anios_cantidad.anio == 1988].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1989].cantidad.values[0] + \
        anios_cantidad[anios_cantidad.anio == 1990].cantidad.values[0]
for index, row in gabriela.iterrows():
    gabriela.set_value(index,'porc_cant', row['cantidad'] / total)

gabriela_top = gabriela.sort_values(['porc_cant'], ascending=[False])
gabriela_top.reset_index(inplace=True)

luciana = nombres_full_top100.loc[2000:2014]
luciana = luciana.groupby(['nombre'])['cantidad'].agg('sum').to_frame()
total = anios_cantidad[(anios_cantidad.anio > 1999) & (anios_cantidad.anio < 2015)].cantidad.sum()
for index, row in luciana.iterrows():
    luciana.set_value(index,'porc_cant', row['cantidad'] / total)

luciana_top = luciana.sort_values(['porc_cant'], ascending=[False])
luciana_top.reset_index(inplace=True)

luciana = nombres_full_top100.loc[(nombres_full_top100.index.get_level_values('anio') > 1999) & \
                                (nombres_full_top100.index.get_level_values('anio') < 2015) & \
                                (nombres_full_top100.index.get_level_values('nombre') == 'luciana')]

for index, row in luciana.iterrows():
    anio = index[0]
    total = anios_cantidad[anios_cantidad.anio == anio].cantidad.values[0]
    luciana.set_value(index,'porc_cant', row['cantidad'] / total)
luciana

del nombres_full_top100

# cruce deportes fin