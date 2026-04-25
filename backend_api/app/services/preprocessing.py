# Canonical text preprocessing module.
# This exact cleaning logic must be used during both model training and live API inference to prevent training-serving skew.
import re
def to_lowercase(text) :
    return text.lower()

def remove_newlines(text) :
    return text.replace('\n', ' ')

def remove_urls(text) :
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, ' ', text)

def remove_html_tags(text) :
    html_pattern = r'<[^>]+>'
    return re.sub(html_pattern, ' ', text)

def remove_extra_punctuation(text) :
    punctuation_pattern = r'([^\w\s])\1{2,}'
    return re.sub(punctuation_pattern, r'\1', text)

def remove_extra_whitespace(text) :
    return " ".join(text.split())

def clean_text(text) :
    return to_lowercase(remove_extra_whitespace(remove_newlines(remove_extra_punctuation(remove_html_tags(remove_urls(text))))))