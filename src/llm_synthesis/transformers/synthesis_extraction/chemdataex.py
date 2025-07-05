import json
from chemdataextractor import Document
import nltk
import os

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# Define a set of common helping (auxiliary) verbs
HELPING_VERBS = {
    'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'
}

def extract_main_verbs(xml_file):
    with open(xml_file, 'rb') as f:
        doc = Document.from_file(f)
    main_verbs = set()
    for sentence in doc.sentences:
        for word, tag in sentence.pos_tagged_tokens:
            if tag.startswith('V'):
                if word.lower() not in HELPING_VERBS:
                    main_verbs.add(word)
    return main_verbs

def get_pii_from_filename(filename):
    """Extracts PII from filename (e.g., filename without extension)."""
    return os.path.splitext(os.path.basename(filename))[0]

if __name__ == '__main__':
    xml_filename = 'S2211379719305820.xml'
    verbs = extract_main_verbs(xml_filename)
    pii = get_pii_from_filename(xml_filename)
    result = {pii: sorted(list(verbs))}
    # Save to JSON
    with open('main_verbs_by_pii.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved main verbs for {pii} to main_verbs_by_pii.json")
