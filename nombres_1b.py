# -*- coding: utf-8 -*-

# Parte 1
# B) Construí un clasificador que separe los nombres en masculinos y femeninos ¿Cuán
# bien funciona? 
# Elegí una métrica apropiada para evaluar el desempeño del clasificador y reportala.
# Para este punto nombre es el nombre completo es decir “Juan Carlos” se considera todo junto.

import pandas as pd
import numpy as np
import time
from math import ceil

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# FUNCIONES #
def limpia_caracteres(lista_caracteres, dataframe, campo):
    for c in lista_caracteres:
        dataframe[campo]  = dataframe[campo].str.replace(c, "")

def limpia_caracteres_con_reemplazo(lista_caracteres, lista_reemplazos, dataframe, campo):
    for i, c in enumerate(lista_caracteres):
        dataframe[campo]  = dataframe[campo].str.replace(c, lista_reemplazos[i])

def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)

        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col+'_enc'] = col_values_transformed

def get_train_test(df, clase_col, ratio):
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
    
    clase_train = df_train[clase_col].values
    clase_test = df_test[clase_col].values
    del df_train[clase_col]
    del df_test[clase_col]

    train = df_train.values
    test = df_test.values
    return train, clase_train, test, clase_test, mask
  
def batch_classify(train, train_clase, test, test_clase, verbose = True):
    cant_classifiers = len(dict_classifiers.keys())

    resultados = pd.DataFrame(data=np.zeros(shape=(cant_classifiers,4)), columns = ['clasificador', 'train_score', 'test_score', 'tiempo_entrenamiento'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.clock()
        classifier.fit(train, train_clase)
        t_end = time.clock()
        t_diff = t_end - t_start
        
        train_score = classifier.score(train, train_clase)
        test_score = classifier.score(test, test_clase)
        resultados.loc[count,'clasificador'] = key
        resultados.loc[count,'train_score'] = train_score
        resultados.loc[count,'test_score'] = test_score
        resultados.loc[count,'tiempo_entrenamiento'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count+=1
    return resultados    

def obtiene_prediccion(datos, campo_salida, evalua_malos = True):
    umbral = 0.75
    prediccion_ori_enc         = clas.predict(datos.iloc[:,2:7])
    prediccion_ori_enc_proba   = clas.predict_proba(datos.iloc[:,2:7])

    campos_originales       = datos.loc[:,['nombre', 'siguiente']]
    campos_originales       = campos_originales.reset_index(drop=True)
    prediccion_ori_enc_df   = pd.DataFrame(prediccion_ori_enc, dtype='str', index=range(0,len(datos)))
    clasificacion           = pd.concat([campos_originales, prediccion_ori_enc_df], axis=1)
    columnas_nombres        = clasificacion.columns.values;
    columnas_nombres[2]     = 'genero_predict_enc'
    clasificacion.columns   = columnas_nombres

    clasificacion[campo_salida]           = np.nan
    clasificacion[campo_salida][clasificacion.genero_predict_enc == '0'] = nombres_genero['genero'][nombres_genero['genero_enc'] == 0].iloc[0]
    clasificacion[campo_salida][clasificacion.genero_predict_enc == '1'] = nombres_genero['genero'][nombres_genero['genero_enc'] == 1].iloc[0]
    clasificacion[campo_salida][clasificacion.genero_predict_enc == '2'] = nombres_genero['genero'][nombres_genero['genero_enc'] == 2].iloc[0]
    clasificacion = clasificacion.drop('genero_predict_enc', 1)

    probabilidad_maxima        = prediccion_ori_enc_proba.max(axis=1)
    clasificacion["proba_max"] = prediccion_ori_enc_proba.max(axis=1)

    # selecciona predicciones "malas" y pone NA
    if evalua_malos:
        clasificacion[campo_salida][(clasificacion["proba_max"] < umbral) & (clasificacion["siguiente"])] = np.nan

    del(campos_originales)
    del(prediccion_ori_enc_df)
    del(prediccion_ori_enc)
    del(prediccion_ori_enc_proba)
    del(probabilidad_maxima)

    return clasificacion

def merge_original_prediccion(df_clasificador_parcial, df_clasificacion, campo):
    # se hace por lotes por memoria
    clasificador_parcial_aux    = []
    inicio                      = 0
    registros                   = len(df_clasificador_parcial)
    lote                        = 50000
    ciclos                      = ceil(registros / lote) + 1
    for i in range(1,ciclos):
        print("Merge i=> "+ str(i))
        fin     = lote * i
        seleccion_lotei   = df_clasificador_parcial[inicio:fin]
        lotei = pd.merge(seleccion_lotei, df_clasificacion, how='left', on=campo)
        clasificador_parcial_aux.append(lotei)
        inicio  = fin
    clasificador = pd.concat(clasificador_parcial_aux, ignore_index=True)

    del(lotei)
    del(clasificador_parcial_aux)
    
    return clasificador
# FUNCIONES FIN #

# datos
directorio          = 'F:/modernizacion/'
directorio_salida   = 'F:/modernizacion/salidas/'

archivo             = 'nombres-permitidos.csv'
path_archivo        = directorio+archivo
data_nombres_genero = pd.read_csv(path_archivo, sep=";")

archivo       = 'historico-nombres.csv'
path_archivo  = directorio+archivo
data_nombres  = pd.read_csv(path_archivo)
# datos fin

# prepara datos
# originales
nombres                     = pd.DataFrame()
nombres                     = data_nombres.loc[:,['nombre', 'nombre']]
columnas_nombres            = nombres.columns.values;
columnas_nombres[1]         = 'nombre_editado'
nombres['nombre']           = nombres['nombre'].astype('str')
nombres['nombre_editado']   = nombres['nombre_editado'].astype('str')

# formateo
nombres['nombre_editado']   = nombres['nombre_editado'].str.strip()
nombres['nombre_editado']   = nombres['nombre_editado'].str.lower()

# sin duplicados / nulos
nombres = nombres.drop_duplicates()
nombres = nombres[~nombres['nombre'].isnull()]

# sin nombres con numeros
nombres_excluidos           = nombres['nombre_editado'].str.contains(r"\d+")
nombres                     = nombres[nombres_excluidos == False]

#nombres         = nombres.reset_index(drop=True)

del(nombres_excluidos)

# limpia caracteres
caracteres = [ '\"', '\/',  ',', '.', '\´', '\`', '(', ')', '-', '\'']
limpia_caracteres(caracteres, nombres, 'nombre_editado')

excluir             = ['á', 'â', 'ä', 'à', 'é', 'ê', 'ë', 'è', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü']
reemplazar          = ['a', 'a', 'a', 'a', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i', 'o', 'o', 'o', 'o', 'o', 'u', 'u', 'u', 'u']
limpia_caracteres_con_reemplazo(excluir, reemplazar, nombres, 'nombre_editado')

# saca conectores
excluir             = [' del ', ' de ', ' los ', ' lo ', ' la ', ' las ']
reemplazar          = [' ', ' ', ' ', ' ', ' ', ' ']
limpia_caracteres_con_reemplazo(excluir, reemplazar, nombres, 'nombre_editado')

# caracter solos
nombres['nombre_editado']  = nombres['nombre_editado'].str.replace(r" [a-z] ", "")

# split nombres
nombres_split = nombres['nombre_editado'].str.split(expand=True)
nombres = pd.concat([nombres, nombres_split], axis=1)
del(nombres_split)

# nombres columnas
cantidad = len(nombres.columns) - 1
columnas = ['nombre', 'nombre_editado']
for i in range(1,cantidad):
    columnas.append('nombre'+str(i))
nombres.columns = columnas    

# datos para clasificador
nombres_genero                 = data_nombres_genero.copy()
nombres_genero                 = nombres_genero.loc[:,['NOMBRE', 'SEXO']]

columnas_nombres    = nombres_genero.columns.values;
columnas_nombres[0] = 'nombre'  
columnas_nombres[1] = 'genero'  
nombres_genero.columns = columnas_nombres

# formateo
nombres_genero['nombre']   = nombres_genero['nombre'].astype('str')
nombres_genero['nombre']   = nombres_genero['nombre'].str.strip()
nombres_genero['nombre']   = nombres_genero['nombre'].str.lower()
nombres_genero['genero']   = nombres_genero['genero'].str.strip()
nombres_genero['genero']   = nombres_genero['genero'].str.lower()

nombres_genero["terminacion"]  = nombres_genero['nombre'].str[-1:]
nombres_genero["terminacion2"] = nombres_genero['nombre'].str[-2:]
nombres_genero["terminacion3"] = nombres_genero['nombre'].str[-3:]
nombres_genero["terminacion4"] = nombres_genero['nombre'].str[-4:]
nombres_genero["siguiente"]    = np.nan # no se usa, sirve para concatenar
nombres_genero = nombres_genero[['nombre', "terminacion", "terminacion2", "terminacion3", "terminacion4", "genero", "siguiente"]]  

# datos originales para clasificar
clasificador_parcial = nombres.loc[:,['nombre', 'nombre_editado', 'nombre1', 'nombre2', 'nombre3']]
clasificador_parcial = clasificador_parcial[(~clasificador_parcial.nombre1.isnull())]

nombres_primero                     = pd.DataFrame(clasificador_parcial.loc[:,['nombre1','nombre2']][~clasificador_parcial['nombre1'].isnull()])
nombres_primero["terminacion"]      = nombres_primero['nombre1'].str[-1:]
nombres_primero["terminacion2"]     = nombres_primero['nombre1'].str[-2:]
nombres_primero["terminacion3"]     = nombres_primero['nombre1'].str[-3:]
nombres_primero["terminacion4"]     = nombres_primero['nombre1'].str[-4:]
nombres_primero["genero"]           = np.nan
nombres_primero["siguiente"]        = (~nombres_primero['nombre2'].isnull())
nombres_primero = nombres_primero.drop('nombre2', 1)
nombres_primero                     = nombres_primero.drop_duplicates()

columnas_nombres    = nombres_primero.columns.values;
columnas_nombres[0] = 'nombre'  
nombres_primero.columns = columnas_nombres

nombres_segundo                     = pd.DataFrame(clasificador_parcial.loc[:,['nombre2','nombre3']][~clasificador_parcial['nombre2'].isnull()])
nombres_segundo["terminacion"]      = nombres_segundo['nombre2'].str[-1:]
nombres_segundo["terminacion2"]     = nombres_segundo['nombre2'].str[-2:]
nombres_segundo["terminacion3"]     = nombres_segundo['nombre2'].str[-3:]
nombres_segundo["terminacion4"]     = nombres_segundo['nombre2'].str[-4:]
nombres_segundo["genero"]           = np.nan
nombres_segundo["siguiente"]        = (~nombres_segundo['nombre3'].isnull())
nombres_segundo = nombres_segundo.drop('nombre3', 1)
nombres_segundo                     = nombres_segundo.drop_duplicates()

columnas_nombres    = nombres_segundo.columns.values;
columnas_nombres[0] = 'nombre'  
nombres_segundo.columns = columnas_nombres

nombres_tercero                     = pd.DataFrame(clasificador_parcial['nombre3'][~clasificador_parcial['nombre3'].isnull()])
nombres_tercero["terminacion"]      = nombres_tercero['nombre3'].str[-1:]
nombres_tercero["terminacion2"]     = nombres_tercero['nombre3'].str[-2:]
nombres_tercero["terminacion3"]     = nombres_tercero['nombre3'].str[-3:]
nombres_tercero["terminacion4"]     = nombres_tercero['nombre3'].str[-4:]
nombres_tercero["genero"]           = np.nan
nombres_tercero["siguiente"]        = np.nan # no se usa, sirve para concatenar
nombres_tercero                     = nombres_tercero.drop_duplicates()

columnas_nombres    = nombres_tercero.columns.values;
columnas_nombres[0] = 'nombre'  
nombres_tercero.columns = columnas_nombres

# unifica todo los datos a usar por el clasificador
# para categorizar todo junto
nombres_a_clasificar = pd.concat([nombres_genero, nombres_primero, nombres_segundo, nombres_tercero], keys=['g', '1', '2', '3'])
nombres_a_clasificar['nombre']   = nombres_a_clasificar['nombre'].astype('str')

# categoriza
# columnas no nulas
columnas = ['nombre', 'terminacion', 'terminacion2', 'terminacion3', 'terminacion4']
label_encode(nombres_a_clasificar, columnas)

# categoriza genero solo en la que existe
nombres_genero  = nombres_a_clasificar.loc['g']
columnas = ['genero']
label_encode(nombres_genero, columnas)

nombres_primero = nombres_a_clasificar.loc['1']
nombres_segundo = nombres_a_clasificar.loc['2']
nombres_tercero = nombres_a_clasificar.loc['3']

del(nombres_a_clasificar)

# evalua clasificadores
clase               = 'genero_enc'
train_test_ratio    = 0.7
train, train_clase, test, test_clase, mask = get_train_test(nombres_genero.iloc[:,7:13], clase, train_test_ratio)

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 18),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB()
}

resultados_clasificadores = batch_classify(train, train_clase, test, test_clase)
resultados_clasificadores
#evalua clasificadores

# Mejor clasificador
# Gradient Boosting Classifier
clas = GradientBoostingClassifier()

# entrena
clas.fit(train, train_clase) 

# scores
train_score = clas.score(train, train_clase)
test_score = clas.score(test, test_clase)
print("Score train => "+ str(train_score))
print("Score test => "+ str(test_score))

# predice test
prediccion_test_enc     = clas.predict(test)
prediccion_test_enc_proba = clas.predict_proba(test)

campos_originales       = nombres_genero.loc[:,['nombre', 'genero']]
campos_originales       = campos_originales[~mask]
campos_originales       = campos_originales.reset_index(drop=True)
prediccion_test_enc_df  = pd.DataFrame(prediccion_test_enc, dtype='str', index=range(0,len(test)))
clasificacion_test      = pd.concat([campos_originales, prediccion_test_enc_df], axis=1)
columnas_nombres        = clasificacion_test.columns.values;
columnas_nombres[2]     = 'genero_predict_enc'
clasificacion_test.columns = columnas_nombres

clasificacion_test["genero_predict"]           = np.nan
clasificacion_test['genero_predict'][clasificacion_test.genero_predict_enc == '0'] = nombres_genero['genero'][nombres_genero['genero_enc'] == 0].iloc[0]
clasificacion_test['genero_predict'][clasificacion_test.genero_predict_enc == '1'] = nombres_genero['genero'][nombres_genero['genero_enc'] == 1].iloc[0]
clasificacion_test['genero_predict'][clasificacion_test.genero_predict_enc == '2'] = nombres_genero['genero'][nombres_genero['genero_enc'] == 2].iloc[0]
clasificacion_test = clasificacion_test.drop('genero_predict_enc', 1)

# confusion matrix
pd.crosstab(test_clase, prediccion_test_enc, rownames=['Generos'], colnames=['Prediccion Generos'])

# guarda prediccion test        
archivo         = 'test_predict.csv'
path_archivo    = directorio_salida+archivo
clasificacion_test.to_csv(path_archivo, encoding='utf-8', index=False)

del(campos_originales)
del(prediccion_test_enc_df)
del(prediccion_test_enc)
del(prediccion_test_enc_proba)
# Mejor clasificador

# predice originales
# primero
campo                   = 'nombre1'
registros               = nombres_primero.loc[:,['nombre', 'siguiente', 'nombre_enc',  'terminacion_enc', 'terminacion2_enc', 'terminacion3_enc', 'terminacion4_enc']]
clasificacion1          = obtiene_prediccion(registros, 'genero_pred_1')

columnas_nombres        = clasificacion1.columns.values;
columnas_nombres[0]     = campo
clasificacion1.columns  = columnas_nombres
 
# agrega prediccion al original
clasificacion1_seleccion_columnas   = clasificacion1.loc[:,['nombre1', 'genero_pred_1']][clasificacion1['siguiente'] == 1]
clasificador_siguiente              = merge_original_prediccion(clasificador_parcial[~clasificador_parcial['nombre2'].isnull()], clasificacion1_seleccion_columnas, campo)
clasificacion1_seleccion_columnas   = clasificacion1.loc[:,['nombre1', 'genero_pred_1']][clasificacion1['siguiente'] == 0]
clasificador_sin_siguiente          = merge_original_prediccion(clasificador_parcial[clasificador_parcial['nombre2'].isnull()], clasificacion1_seleccion_columnas, campo)
clasificador_parcial                = pd.concat([clasificador_siguiente, clasificador_sin_siguiente], ignore_index=True)

del(clasificacion1_seleccion_columnas)
del(clasificador_siguiente)
del(clasificador_sin_siguiente)
del(registros)
del(nombres_primero)
# primero

# segundo
campo                   = 'nombre2'
registros               = nombres_segundo.loc[:,['nombre', 'siguiente', 'nombre_enc',  'terminacion_enc', 'terminacion2_enc', 'terminacion3_enc', 'terminacion4_enc']]
clasificacion2          = obtiene_prediccion(registros, 'genero_pred_2')

columnas_nombres        = clasificacion2.columns.values;
columnas_nombres[0]     = campo
clasificacion2.columns  = columnas_nombres
 
# agrega prediccion al original
clasificacion2_seleccion_columnas   = clasificacion2.loc[:,['nombre2', 'genero_pred_2']][clasificacion2['siguiente'] == 1]
clasificador_siguiente              = merge_original_prediccion(clasificador_parcial[~clasificador_parcial['nombre3'].isnull()], clasificacion2_seleccion_columnas, campo)
clasificacion2_seleccion_columnas   = clasificacion2.loc[:,['nombre2', 'genero_pred_2']][clasificacion2['siguiente'] == 0]
clasificador_sin_siguiente          = merge_original_prediccion(clasificador_parcial[clasificador_parcial['nombre3'].isnull()], clasificacion2_seleccion_columnas, campo)
clasificador_parcial                = pd.concat([clasificador_siguiente, clasificador_sin_siguiente], ignore_index=True)

del(clasificacion2_seleccion_columnas)
del(clasificador_siguiente)
del(clasificador_sin_siguiente)
del(registros)
del(nombres_segundo)
# segundo

# tercero
campo                   = 'nombre3'
registros               = nombres_tercero.loc[:,['nombre', 'siguiente', 'nombre_enc',  'terminacion_enc', 'terminacion2_enc', 'terminacion3_enc', 'terminacion4_enc']]
clasificacion3          = obtiene_prediccion(registros, 'genero_pred_3', False)

columnas_nombres        = clasificacion3.columns.values;
columnas_nombres[0]     = 'nombre3'
clasificacion3.columns  = columnas_nombres
 
clasificacion3_seleccion_columnas   = clasificacion3.loc[:,['nombre3', 'genero_pred_3']]
clasificador_parcial                = merge_original_prediccion(clasificador_parcial, clasificacion3_seleccion_columnas, campo)

del(clasificacion3_seleccion_columnas)
del(registros)
del(nombres_tercero)
# tercero

# genero clasificacion final
clasificador_parcial["genero_pred"]           = np.nan
clasificador_parcial['genero_pred'][(clasificador_parcial.genero_pred.isnull()) & \
                                        (~clasificador_parcial.genero_pred_1.isnull()) & \
                                        (clasificador_parcial.genero_pred_1 != 'a') \
                                        ]  = clasificador_parcial.genero_pred_1
clasificador_parcial['genero_pred'][(clasificador_parcial.genero_pred.isnull()) & \
                                        ((clasificador_parcial.genero_pred_1.isnull()) | (clasificador_parcial.genero_pred_1 == 'a')) & \
                                        (~clasificador_parcial.genero_pred_2.isnull()) & (clasificador_parcial.genero_pred_2 != 'a') \
                                        ]  = clasificador_parcial.genero_pred_2
clasificador_parcial['genero_pred'][(clasificador_parcial.genero_pred.isnull()) & \
                                        (clasificador_parcial.genero_pred_1 == 'a') & \
                                        ((clasificador_parcial.genero_pred_2.isnull()) | (clasificador_parcial.genero_pred_2 == 'a')) & \
                                        (~clasificador_parcial.genero_pred_3.isnull()) \
                                        ]  = clasificador_parcial.genero_pred_3
clasificador_parcial['genero_pred'][(clasificador_parcial.genero_pred.isnull()) & \
                                        (clasificador_parcial.genero_pred_1 == 'a') & \
                                        ((clasificador_parcial.genero_pred_2.isnull()) | (clasificador_parcial.genero_pred_2 == 'a')) & \
                                        (clasificador_parcial.genero_pred_3.isnull()) \
                                        ]  = clasificador_parcial.genero_pred_1
clasificador_parcial['genero_pred'][(clasificador_parcial.genero_pred.isnull()) & \
                                        (clasificador_parcial.genero_pred_1.isnull()) & \
                                        ((clasificador_parcial.genero_pred_2.isnull()) | (clasificador_parcial.genero_pred_2 == 'a')) & \
                                        (~clasificador_parcial.genero_pred_3.isnull()) \
                                        ]  = clasificador_parcial.genero_pred_3
clasificador_parcial['genero_pred'][(clasificador_parcial.genero_pred.isnull()) & \
                                        (clasificador_parcial.genero_pred_1.isnull()) & \
                                        ((clasificador_parcial.genero_pred_2.isnull()) | (clasificador_parcial.genero_pred_2 == 'a')) & \
                                        (clasificador_parcial.genero_pred_3.isnull()) \
                                        ]  = clasificador_parcial.genero_pred_2
                    
# fecuencias
nombres_genero['genero'].value_counts()
unique_elements, counts_elements = np.unique(train_clase, return_counts=True)
unique_elements
counts_elements
unique_elements, counts_elements = np.unique(test_clase, return_counts=True)
unique_elements
counts_elements
clasificador_parcial['genero_pred'].value_counts()

# guarda prediccion parcial        
archivo         = 'original_predict_parcial.csv'
path_archivo    = directorio_salida+archivo
clasificador_parcial.to_csv(path_archivo, encoding='utf-8', index=False)

# genero salida final
clasificador_parcial_seleccion_columnas = clasificador_parcial.loc[:,['nombre', 'genero_pred']]
resultado = pd.merge(data_nombres, clasificador_parcial_seleccion_columnas, how='left', on='nombre')

# guarda prediccion        
archivo         = 'original_predict.csv'
path_archivo    = directorio_salida+archivo
resultado.to_csv(path_archivo, encoding='utf-8', index=False)

del(clasificador_parcial_seleccion_columnas)
del(clasificador_parcial)
del(resultado)
