import math
import os
import re
from collections import defaultdict, Counter

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) #every set of letters bound by two whitespaces is counted as a word/token

def soundex(word):
    word = word.upper()
    codes = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3",
             "L": "4", "MN": "5", "R": "6"} #simple soundex algorithm implementation
    soundex_code = word[0]

    for char in word[1:]:
        for key in codes:
            if char in key:
                code = codes[key]
                if code != soundex_code[-1]:
                    soundex_code += code
                break
        else:
            if soundex_code[-1] != "0":
                soundex_code += "0"

    soundex_code = soundex_code.replace("0", "")
    return (soundex_code + "000")[:4]

def build_index(corpus_dir):
    dictionary = defaultdict(list) # term → postings [(docID, tf)]
    doc_lengths = defaultdict(float) # docID → |doc|
    N = 0 # total docs

    for filename in os.listdir(corpus_dir):
        path = os.path.join(corpus_dir, filename)
        if not os.path.isfile(path):
            continue
        N += 1
        with open(path, "r", encoding="utf-8") as f:
            tokens = tokenize(f.read())
            freqs = Counter(tokens)
            for term, tf in freqs.items():
                dictionary[term].append((filename, tf))


    for term, postings in dictionary.items(): # compute normalized lengths (lnc)
        for docID, tf in postings:
            wdt = 1 + math.log10(tf)
            doc_lengths[docID] += wdt ** 2

    for docID in doc_lengths:
        doc_lengths[docID] = math.sqrt(doc_lengths[docID])

    return dictionary, doc_lengths, N

def build_query_vector(query, dictionary, N):
    tokens = tokenize(query)
    freqs = Counter(tokens)
    qvec = {}
    for term, tf in freqs.items():
        if term in dictionary:
            df = len(dictionary[term])
            idf = math.log10(N / df)
            qvec[term] = (1 + math.log10(tf)) * idf # normalize query
    norm = math.sqrt(sum(w ** 2 for w in qvec.values()))
    if norm > 0:
        qvec = {t: w / norm for t, w in qvec.items()}
    return qvec

def search(query, dictionary, doc_lengths, N, top_k=10):
    qvec = build_query_vector(query, dictionary, N)
    scores = defaultdict(float)

    for term, wq in qvec.items():
        if term not in dictionary: # soundex fallback
            q_code = soundex(term)
            for dict_term in dictionary.keys():
                if soundex(dict_term) == q_code:
                    for docID, tf in dictionary[dict_term]:
                        wdt = 1 + math.log10(tf)
                        scores[docID] += wq * (wdt / doc_lengths[docID])
        else:
            for docID, tf in dictionary[term]:
                wdt = 1 + math.log10(tf)
                scores[docID] += wq * (wdt / doc_lengths[docID])

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]

def save_postings(dictionary, filepath="postings.txt"): #saving the postings as a separate text file
    with open(filepath, "w", encoding="utf-8") as f:
        for term in sorted(dictionary.keys()):
            postings = dictionary[term]
            df = len(postings)
            postings_str = " -> ".join(f"({docID},{tf})" for docID, tf in postings)
            f.write(f"{term} {df} -> {postings_str}\n")


corpus_dir = r"C:\Users\venka\Downloads\inforetlab1\Corpus"  #when running your code replace this with your corpus path
dictionary, doc_lengths, N = build_index(corpus_dir)

save_postings(dictionary, "postings.txt")
query = "Warwickshire, came from an ancient family and was the heiress to some land"
results = search(query, dictionary, doc_lengths, N)

print("Top results for:", query)
output = [(docID, score) for docID, score in results]
print(output)
