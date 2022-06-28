import pickle
class Clasificador:

    def __init__(self,idioma):
        if idioma == "ES":
            self.model = pickle.load(open('model_ES.pkl', 'rb'))
        if idioma == "EN":
            self.model = pickle.load(open('model.pkl', 'rb'))

    def predecir(self,test_input_ids,test_attention_masks):
        return self.model.predict([test_input_ids,test_attention_masks])


    