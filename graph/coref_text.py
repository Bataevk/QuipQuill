import spacy
import coreferee

def resolve_coreferences(text: str) -> str:
    # Загружаем spaCy-трансформер версии ≥3.7.2
    nlp = spacy.load("en_core_web_trf")
    # Вставляем Coreferee в pipeline
    nlp.add_pipe("coreferee")

    doc = nlp(text)
    tokens = []
    for token in doc:
        resolved = doc._.coref_chains.resolve(token)
        tokens.append(" ".join([t.text for t in resolved]) if resolved else token.text)
    return " ".join(tokens)

if __name__ == "__main__":
    sample = ("Although he was very busy, Peter had had enough of it. "
              "He and his wife decided on a holiday.")
    print(resolve_coreferences(sample))
