import torch
from transformers import BertModel, BertTokenizer
from pypdf import PdfReader

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + ' '
    return text

def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden state as the document embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding


if __name__ == "__main__":
    pdf_path = 'document.pdf'  
    text = pdf_to_text(pdf_path)
    embedding = text_to_embedding(text)
    print(embedding)
