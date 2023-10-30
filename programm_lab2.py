import docx
import PyPDF2
import re


docx_file_path = 'Фрагмент.docx'
pdf_file_path = 'Фрагмент.pdf'


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Zа-яА-Я\- ]', '', text)
    text = text.lower()
    words = text.split()
    return words


def count_words(words):
    total_words = len(words)
    unique_words = len(set(words))
    return total_words, unique_words


def load_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)


def load_text_from_pdf(file_path):
    text = ''
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    
    pdf_file.close()
    return text


def extract_short_words(words):
    short_words = []
    for word in words:
        if len(word) >= 1 and len(word) <= 2:
            short_words.append(word)
    return short_words


def start():
    doc_text = load_text_from_docx(docx_file_path)
    pdf_text = load_text_from_pdf(pdf_file_path)
    doc_words_preprocessed = preprocess_text(doc_text)
    pdf_words_preprocessed = preprocess_text(pdf_text)
    print(count_words(doc_words_preprocessed))
    print(count_words(pdf_words_preprocessed))
    short_words_doc = extract_short_words(doc_words_preprocessed)
    short_words_pfd = extract_short_words(pdf_words_preprocessed)
    print(len(short_words_doc))
    print(len(short_words_pfd))
    

if __name__ == '__main__':
    start()