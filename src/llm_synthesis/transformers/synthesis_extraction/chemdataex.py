# Standard library imports
import json
import os

# Third-party imports
from chemdataextractor.doc import Document
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# Download required NLTK data if not already present
nltk.download('stopwords')
nltk.download('wordnet')

# Define helping verbs (auxiliaries)
HELPING_VERBS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could"
}

STOP_WORDS = set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS tag for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def extract_main_verbs(filepath):
    from chemdataextractor.doc import Document
    lemmatizer = WordNetLemmatizer()
    with open(filepath, 'rb') as f:
        doc = Document.from_file(f)
    main_verbs = set()
    for sentence in doc.sentences:
        for word, tag in sentence.pos_tagged_tokens:
            if tag.startswith('V'):
                w_lower = word.lower()
                # Remove helping verbs and stop words
                if w_lower not in HELPING_VERBS and w_lower not in STOP_WORDS:
                    lemma = lemmatizer.lemmatize(w_lower, get_wordnet_pos(tag))
                    main_verbs.add(lemma)
    return main_verbs


if __name__ == '__main__':
    file_path = '10.26434:chemrxiv.12581321.v1.pdf'
    verbs = extract_main_verbs(file_path)
    # Use the filename (without path) as the key
    file_key = os.path.basename(file_path)
    result = {file_key: sorted(list(verbs))}
    # Save to JSON
    with open('main_verbs_by_filename.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
