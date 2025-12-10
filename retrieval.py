import varint
import json
import time
import struct
import bbhash
import hashlib
from nltk.stem.porter import PorterStemmer
from utils.tokenize import tokenize_text

stemmer = PorterStemmer()

with open('/Users/arnavgosain/Documents/cs121/mapping_lexicon.json', 'r') as lf:
    MAPPING_LEXICON = json.load(lf)

mphf_path = '/Users/arnavgosain/Documents/cs121/lexicon_merged.mphf'
offsets_path = '/Users/arnavgosain/Documents/cs121/offsets.bin'

LEXICON_MPHF = bbhash.PyMPHF([0], 1, 1.0, 0)
LEXICON_MPHF.load(mphf_path)

with open(offsets_path, 'rb') as f:
    OFFSETS_DATA = f.read()

def deterministic_hash(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

def get_offset(word):
    hash_val = deterministic_hash(word) & 0xFFFFFFFFFFFFFFFF
    hash_idx = LEXICON_MPHF.lookup(hash_val)
    if hash_idx is None:
        raise KeyError(f"Word '{word}' not found in lexicon")
    offset_bytes = OFFSETS_DATA[hash_idx * 8 : (hash_idx + 1) * 8]
    return struct.unpack('<Q', offset_bytes)[0]

def load_words(bin_path, words):
    results = {}

    with open(bin_path, "rb") as f:
        for word in words:
            try:
                offset = get_offset(word)
            except KeyError:
                results[word] = []
                continue

            # Seek to start of term block
            f.seek(offset)

            # ---- Read term ----
            term_len = varint.decode_stream(f)
            term_bytes = f.read(term_len)
            term = term_bytes.decode("utf-8")

            # (Optional sanity check)
            assert term == word, "Lexicon offset mismatch!"

            # ---- Read postings count  ----
            postings_count = varint.decode_stream(f)

            # ---- Decode postings ----
            postings = []
            last_id = 0

            for _ in range(postings_count):
                delta = varint.decode_stream(f)
                score = varint.decode_stream(f)

                doc_id = last_id + delta
                last_id = doc_id
                tfidf = round(score / 10000, 4)

                postings.append((doc_id, tfidf))

            results[word] = postings

    return results


def intersect_two(p1, p2):
    i = j = 0
    intersection = []

    while i < len(p1) and j < len(p2):
        d1, score1 = p1[i]
        d2, score2 = p2[j]

        if d1 == d2:
            intersection.append((d1, score1+score2))
            i += 1
            j += 1
        elif d1 < d2:
            i += 1
        else:
            j += 1
    return intersection


def intersect_k(lists):
    if not lists:
        return []

    lists.sort(key = len)

    result = lists[0]
    for lst in lists[1:]:
        result = intersect_two(result, lst)
        if not result:
            break
    return result


def shortid_to_document(doc: tuple[int, float]):
    docid, score = doc
    key = str(docid)

    if key not in MAPPING_LEXICON:
        return None

    offset = MAPPING_LEXICON[key]

    with open('/Users/arnavgosain/Documents/cs121/mapping.json', 'rb') as f:
        f.seek(offset)
        line = f.readline()
        data = json.loads(line.decode('utf-8'))
        data[key].append(score)
        return data[key]

def search(phrase: str):
    """Search for documents matching the given phrase"""
    start_time = time.time()
    tokens, _ = tokenize_text(phrase)
    query_words = [stemmer.stem(tok) for tok in tokens.keys()]
    print(query_words)
    bin_path = '/Users/arnavgosain/Documents/cs121/inverted_index_merged.bin'

    terms = load_words(bin_path, query_words)
    docs = intersect_k(list(terms.values()))
    docs = sorted([shortid_to_document(d) for d in docs], key=lambda x: x[2], reverse=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    return docs, elapsed_time

if __name__ == '__main__':
    phrase = input("enter phrase: ")
    docs, elapsed_time = search(phrase)

    print(f"Search completed in {elapsed_time:.4f} seconds")
    print(f"Results: {len(docs)}")
    print(docs)
