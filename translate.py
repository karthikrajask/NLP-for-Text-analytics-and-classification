import pandas as pd
from transformers import MBartForConditionalGeneration, MBartTokenizer
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def translate(text):
    try:
        detected_lang = detect(text)

        lang_mapping = {
            'en': 'en_XX',  
            'hi': 'hi_IN',  
            'es': 'es_XX',  
            'fr': 'fr_XX',  
            'de': 'de_DE',  
            'ja': 'ja_XX',  
            
        }

        if detected_lang in lang_mapping:
            source_lang = lang_mapping[detected_lang]
        else:
            return f"Language '{detected_lang}' is not supported for translation."
       
        target_lang = 'en_XX'

        inputs = tokenizer(text, return_tensors="pt", src_lang=source_lang)

        translated_tokens = model.generate(inputs["input_ids"], forced_bos_token_id=tokenizer.lang2id[target_lang])

        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    except Exception as e:
        return str(e)

input_file_path = "train.csv" 
df = pd.read_csv(input_file_path)

if 'crimeaditionalinfo' not in df.columns:
    raise ValueError("CSV must contain a 'text' column.")

df['translated_text'] = df['crimeaditionalinfo'].apply(translate)


output_file_path = "translated_texts.csv"
df.to_csv(output_file_path, index=False)

print(f"Translations saved to {output_file_path}.")

