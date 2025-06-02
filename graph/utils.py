import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer  
import string
import re

import yaml
import logging
from typing import Dict, Any


nltk.download('punkt')
nltk.download('stopwords')

# Инициализируем лемматизатор
__lemmatizer = WordNetLemmatizer() 

# Скачиваем необходимые ресурсы при первом использовани (для лемматизации на английском) 
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('wordnet')  

# def preprocess_text(text):
#     # Приведение к нижнему регистру
#     text = text.lower()
#     # Удаление знаков препинания
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Токенизация
#     tokens = word_tokenize(text)
#     # Удаление стоп-слов
#     stop_words = set(stopwords.words('english'))  # Замените 'english' на нужный язык
#     tokens = [word for word in tokens if word not in stop_words]
#     # Стемминг
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(word) for word in tokens]
#     return ' '.join(tokens)

def lemmatize_phrase(phrase):  
    """
    Лемматизирует фразу, заменяя слова на их начальную форму (лемму).
    """
    words = nltk.word_tokenize(phrase.lower())  # Токенизация фразы на отдельные слова  
    lemmatized_words = [__lemmatizer.lemmatize(word, pos='n') for word in words]  
    return ' '.join(lemmatized_words)  # Объединяем обратно в строку  


def preprocess_text(text: str) -> str:
    """Преобразует текст, убирая лишние пробелы, точки и символы. Затем выполняет лемматизацию"""
    # замена '-' на пробел
    text = text.replace('-', ' ').lower()
    # Убираем токены вида <|КАКОЙ_ТО ТЕКСТ|>
    text = remove_tokens(text)
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Удаление знаков препинания
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Убираем символы, которые не нужны
    text = re.sub(r'[^\w\s]', '', text)

    # Удаляем артикли
    text = re.sub(r'\b(?:a|an|the)\b', '', text)

    # Лемматизируем текст
    text = lemmatize_phrase(text)

    return text


def remove_tokens(text):  
  """Удаляет токены вида <|КАКОЙ_ТО ТЕКСТ|> из строки.  

  Args:  
    text: Строка, из которой нужно удалить токены.  

  Returns:  
    Строка, из которой удалены токены.  
  """  
  if not text or text.strip() == "":
    return ""  # Возвращаем пустую строку, если входная строка пустая
  pattern = r"<\|[^|>]*\|>"  
  return re.sub(pattern, "", text, flags=re.IGNORECASE)  

def load_config(config_path: str) -> Dict[str, Any]:
  """Загружает конфигурацию из YAML файла."""
  logging.info(f"Загрузка конфигурации из файла: {config_path}")
  try:
      with open(config_path, 'r', encoding='utf-8') as f:
          config = yaml.safe_load(f)
      if not isinstance(config, dict):
            raise ValueError(f"Файл конфигурации {config_path} не содержит валидный YAML словарь.")
      logging.info(f"Конфигурация успешно загружена.")
      # TODO: Добавить валидацию наличия необходимых ключей в config (например, llm, prompts и т.д.)
      return config
  except FileNotFoundError:
      logging.error(f"Файл конфигурации не найден: {config_path}")
      raise # Передаем исключение выше
  except yaml.YAMLError as e:
      logging.error(f"Ошибка парсинга YAML файла {config_path}: {e}")
      raise ValueError(f"Некорректный формат YAML в файле {config_path}") from e
  except Exception as e:
      logging.error(f"Неожиданная ошибка при загрузке config файла {config_path}: {e}", exc_info=True)
      raise ValueError(f"Ошибка при чтении файла конфигурации {config_path}") from e
