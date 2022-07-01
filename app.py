import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, send_file, send_from_directory, redirect

from transformers import AutoTokenizer, AutoModel, TFAutoModel,TFRobertaForSequenceClassification, TFBertModel, TFRobertaModel
from flask_assets import Environment, Bundle
from modelo import Modelo

app = Flask(__name__)

modelo = Modelo("ES")


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/fakeNewsIngles')
def ingles():
    #return 'Hello World'
    
    return render_template('ingles.html')

@app.route('/fakeNewsEspañol')
def español():
    # return 'Hello World'

    return render_template('español.html')
    # return render_template('index.html')

# Revisar: Hacer un predict ingles y otro en español
@app.route('/predictIngles',methods = ['POST'])
def predictIngles():
    test_texts = [x for x in request.form.values()]
    if len(test_texts[0]) == 0:
        return render_template('resultIngles.html', prediction_text="Debe de introducir al menos 1 palabra")
    prediccion = modelo.predecir(test_texts)
    return render_template('resultIngles.html', prediction_text="La noticia es: {} ".format(prediccion))

@app.route('/predictEspañol',methods = ['POST'])
def predictEspañol():
    test_texts = [x for x in request.form.values()]
    if len(test_texts[0]) == 0:
        return render_template('resultEspañol.html', prediction_text="Debe de introducir al menos 1 palabra")
    prediccion = modelo.predecir(test_texts)
    return render_template('resultEspañol.html', prediction_text="La noticia es: {} ".format(prediccion))


@app.route('/return-files', methods=['GET'])
def download_ingles():
    return send_file( as_attachment=True, path_or_file="model.pkl")


if __name__ == '__main__':
    app.run(debug=False, port=8080)