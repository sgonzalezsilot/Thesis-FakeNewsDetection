from clasificador import Clasificador
from preprocesador import Preprocesador
import numpy as np

class Modelo:

    def __init__(self,idioma):
        self.clasificador = Clasificador(idioma)
        self.preprocesador = Preprocesador(idioma)
    
    def predecir(self,texto):
        test_input_ids,test_attention_masks = self.prepocesador.prepocesar(texto)
        prediccion = self.clasificador.predecir(test_input_ids,test_attention_masks)
        prediccion = np.round(prediccion)
        print(prediccion)

        if prediccion == 0:
            output = "Verdadera"
        else:
            output = "Falsa"

        return output
        