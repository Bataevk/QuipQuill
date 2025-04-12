import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer  
import string
import re

nltk.download('punkt')
nltk.download('stopwords')

# Инициализируем лемматизатор
__lemmatizer = WordNetLemmatizer() 

# Скачиваем необходимые ресурсы при первом использовани (для лемматизации на английском) 
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('wordnet')  

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление знаков препинания
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Токенизация
    tokens = word_tokenize(text)
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))  # Замените 'english' на нужный язык
    tokens = [word for word in tokens if word not in stop_words]
    # Стемминг
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
