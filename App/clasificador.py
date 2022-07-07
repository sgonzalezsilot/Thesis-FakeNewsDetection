import pickle
class Clasificador:

    def __init__(self,idioma):
        if idioma == "ES":
            self.modelo = pickle.load(open('models/RobertaBIO_Adamax.pkl', 'rb'))
        if idioma == "EN":
            self.modelo = pickle.load(open("models/V2_Adamaxpre.pkl", 'rb'))

    def predecir(self,test_input_ids,test_attention_masks):
        return self.modelo.predict([test_input_ids,test_attention_masks])


