from transformers import AutoTokenizer, AutoModel, TFAutoModel,TFRobertaForSequenceClassification, TFBertModel, TFRobertaModel
import numpy as np

class Preprocesador:

    def __init__(self,idioma):
        if idioma == "ES":
            self.sentence_length = 96
            self.model_name = "cardiffnlp/twitter-roberta-base-mar2022"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif idioma == "EN":
            self.sentence_length = 310
            # Revisar: Actualizar
            self.model_name = "cardiffnlp/twitter-roberta-base-mar2022"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


    def bert_encode(self,data,maximum_length):
        input_ids = []
        attention_masks = []

        for i in range(len(data)):
            encoded = self.tokenizer.encode_plus(
                
                data[i],
                add_special_tokens=True,
                max_length=maximum_length,
                pad_to_max_length=True,
                truncation = True,
                return_attention_mask=True,
            )
            
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids),np.array(attention_masks)

    def preprocesar(self,texto):
        test_input_ids,test_attention_masks = self.bert_encode(texto,self.sentence_length)
        return test_input_ids,test_attention_masks
