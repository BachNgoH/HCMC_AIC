from transformers import MarianTokenizer, MarianMTModel

src = "vi"
trg = "en"

model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    batch = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(**batch)
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return result
