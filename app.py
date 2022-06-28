import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, send_file, send_from_directory, redirect

from transformers import AutoTokenizer, AutoModel, TFAutoModel, TFRobertaForSequenceClassification, TFBertModel, \
    TFRobertaModel
from flask_assets import Environment, Bundle
from modelo import Modelo

app = Flask(__name__)

# model = pickle.load(open('model.pkl','rb'))

modelo = Modelo("EN")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/fakeNewsIngles')
def ingles():
    # return 'Hello World'

    return render_template('ingles.html')
    # return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    test_texts = [x for x in request.form.values()]
    modelo.predecir(test_texts)
    return render_template('result.html', prediction_text="La noticia es: {} ".format(prediction))


@app.route('/return-files', methods=['GET'])
def download():
    return send_file(as_attachment=True, path_or_file="model.pkl")


if __name__ == '__main__':
    app.run(debug=False, port=80)