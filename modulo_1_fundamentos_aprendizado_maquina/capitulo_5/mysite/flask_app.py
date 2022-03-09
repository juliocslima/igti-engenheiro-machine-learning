
# A very simple Flask Hello World app for you to get started with...

import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import sys


lista_media=[5.86416667, 3.08833333, 3.79583333, 1.21916667]
lista_desvio=[0.8403169871476411, 0.41396121664823726, 1.7801870610197745, 0.7778491534710024]


app = Flask(__name__,template_folder="./templates")


def ajustando_entradas(lista_entradas,lista_media,lista_desvio):

    #z= (x-u)/s

    lista_ajustada=[]
    for (x, u, s) in zip(lista_entradas, lista_media, lista_desvio):

        z=(x-u)/s
        lista_ajustada.append(z)


    return lista_ajustada



def previsao_iris(lista_valores_formulario):
    prever=np.array(lista_valores_formulario).reshape(1,4)      #transforma os valores do formulario
    modelo_salvo = joblib.load('./modelo_knn.joblib')             #realiza a carga do modelo salvo
    resultado = modelo_salvo.predict(prever)                    #aplica a previsao

    return resultado[0]

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/classifica_iris',methods=['POST','GET'])
def result():

    slength=request.args.get("slength")
    swidth=request.args.get("swidth")
    plength=request.args.get("plength")
    pwidth=request.args.get("pwidth")

    lista_recebida=[float(slength),float(swidth),float(plength),float(pwidth)]

    lista_ajustada=ajustando_entradas(lista_recebida,lista_media,lista_desvio)



    classe=previsao_iris(lista_ajustada)
    if int(classe)==0:
        previsao='Setosa'
    elif int(classe)==1:
        previsao='Versicolor'
    else:
        previsao="Virginica"

    data = { 
    	'previsao': previsao,
    	'date': '2022-01-02'
    }

    #retorna o resultado para uma pagina html
    # return render_template("resultado.html", previsao=previsao)
    return jsonify(data), 200








