from flask import Flask, request, render_template, send_file
from modelo import Modelo

app = Flask(__name__)

modelo_EN = Modelo("EN")
modelo_ES = Modelo("ES")


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/fakeNewsIngles')
def ingles():
    return render_template('ingles.html')

@app.route('/fakeNewsEspañol')
def español():
    return render_template('español.html')

@app.route('/predictIngles',methods = ['POST'])
def predictIngles():
    test_texts = [x for x in request.form.values()]
    if len(test_texts[0]) == 0:
        return render_template('resultIngles.html', prediction_text="Debe de introducir al menos 1 palabra")
    prediccion = modelo_EN.predecir(test_texts)
    return render_template('resultIngles.html', prediction_text="La noticia es: {} ".format(prediccion))

@app.route('/predictEspañol',methods = ['POST'])
def predictEspañol():
    test_texts = [x for x in request.form.values()]
    if len(test_texts[0]) == 0:
        return render_template('resultEspañol.html', prediction_text="Debe de introducir al menos 1 palabra")
    prediccion = modelo_ES.predecir(test_texts)
    return render_template('resultEspañol.html', prediction_text="La noticia es: {} ".format(prediccion))


@app.route('/return-files', methods=['GET'])
def download_ingles():
    return send_file(as_attachment=True, path_or_file="models/V2_Adamax.pkl")

@app.route('/return-files', methods=['GET'])
def download_español():
    return send_file(as_attachment=True, path_or_file="models/RobertaBIO_Adamax.pkl")


if __name__ == '__main__':
    app.run(debug=False, port=8080)