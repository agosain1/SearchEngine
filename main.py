import os
import json
from bs4 import BeautifulSoup
from utils.tokenize import tokenize_text
from collections import defaultdict, deque
import datetime
import varint
from nltk.stem import PorterStemmer

ROOT_PATH = '/Users/arnavgosain/Documents/cs121/'
MAX_PROCESS_SIZE = 15000
STOP_LIMIT = 100000

stemmer = PorterStemmer()

def extract_weighted_tokens(soup):
    weights = {
        'h1': 3.0,
        'h2': 2.5,
        'h3': 2.0,
        'strong': 2.0,
        'b': 2.0
    }

    token_freq = defaultdict(float)
    total_count = 0

    # 1. Weighted tags
    for tag, w in weights.items():
        for elem in soup.find_all(tag):
            raw = elem.get_text(" ", strip=True)
            tokens, count = tokenize_text(raw)
            for tok, freq in tokens.items():
                stemmed = stemmer.stem(tok)
                token_freq[stemmed] += freq * w
            total_count += count

            elem.extract()

    # 2. Normal body text
    raw_body = soup.get_text(" ", strip=True)
    body_tokens, body_count = tokenize_text(raw_body)
    for tok, freq in body_tokens.items():
        stemmed = stemmer.stem(tok)
        token_freq[stemmed] += freq

    return token_freq, total_count + body_count


class Posting:
    def __init__(self, short_id, score):
        self.short_id = short_id
        self.score = score

    def __repr__(self):
        return f"[{self.short_id}, {self.score}]"

class Mapping: # map ID: Posting
    def __init__(self):
        self.hsh = dict()

    def add_document(self, cur_id: int, docid: str, url: str):
        if cur_id in self.hsh:
            raise Exception('Duplicate posting')
        if not docid:
            raise Exception('No posting')
        if not url:
            raise Exception('No url')
        self.hsh[cur_id] = (docid, url)

    def offload(self):
        lexicon = {}
        offset = 0

        with open(ROOT_PATH + 'mapping.json', "wb") as f:
            for key, value in self.hsh.items():
                line = json.dumps({key: value}) + b"\n" if isinstance(value, bytes) else (
                            json.dumps({key: value}) + "\n").encode("utf-8")

                lexicon[key] = offset

                f.write(line)
                offset += len(line)

        with open(ROOT_PATH + 'mapping_lexicon.json', "w") as lf:
            json.dump(lexicon, lf)

        self._save_stats()

    def _save_stats(self):
        with open('stats.txt', 'a') as f:
            amount_of_docs = len(self.hsh)
            f.write('[LOG] ' + str(datetime.datetime.now()) + ' Amount of docs: ' + str(amount_of_docs) + '\n')

class InvertedIndex:
    def __init__(self, num):
        self.hsh = defaultdict(list)
        self.num = num
        self.df = defaultdict(int)

    def update(self, token: str, short_id: int, score: float): # sorts while inserting
        postings = self.hsh[token]
        if not postings or postings[-1].short_id != short_id:
            self.df[token] += 1
        postings.append(Posting(short_id, score))

    def offload(self):
        out_path = ROOT_PATH + f"inverted_index{self.num}.bin"
        lex_path = ROOT_PATH + f"lexicon{self.num}.json"

        lexicon = {}  # term -> (offset, postings_count)
        offset = 0  # current byte position

        with open(out_path, "wb") as f:
            for term, postings in self.hsh.items():

                # Record where *this term* begins in the file
                term_offset = offset
                lexicon[term] = term_offset

                term_bytes = term.encode("utf-8")

                # --- Write term length ---
                b = varint.encode(len(term_bytes))
                f.write(b)
                offset += len(b)

                # --- Write term ---
                f.write(term_bytes)
                offset += len(term_bytes)

                # --- Write postings count ---
                b = varint.encode(len(postings))
                f.write(b)
                offset += len(b)

                # --- Delta encode docIDs ---
                last_id = 0
                for p in postings:
                    delta = p.short_id - last_id
                    last_id = p.short_id

                    # write delta
                    b = varint.encode(delta)
                    f.write(b)
                    offset += len(b)

                    # write freq (convert float to int by scaling)
                    score_int = int(round(p.score * 10000))
                    b = varint.encode(score_int)
                    f.write(b)
                    offset += len(b)

        # Save lexicon as JSON
        with open(lex_path, "w") as lf:
            json.dump(lexicon, lf)

        self._save_stats()

    def _save_stats(self):
        with open('stats.txt', 'a') as f:
            f.write(
                f"[LOG] {datetime.datetime.now()} "
                f"Compressed terms: {len(self.hsh)}\n"
            )

class Worker:
    def __init__(self):
        self.mapping = Mapping()
        self.iterator = os.walk(ROOT_PATH + 'DEV')
        self.CUR_ID = 1
        self.subdirs = next(self.iterator)[1]
        self.inverted_index = InvertedIndex(1) # initial number = 1
        self.q = deque()

    def _process_file(self, root: str, file_path: str):
        data = json.load(open(root + '/' + file_path, 'r'))
        url = data['url']
        docid = file_path.split('.')[0]
        self.mapping.add_document(self.CUR_ID, docid, url)

        soup = BeautifulSoup(data['content'], 'lxml')

        tokens, total_count = extract_weighted_tokens(soup)
        if total_count == 0:
            return

        for token, freq in tokens.items():
            tf = freq / total_count
            self.inverted_index.update(token, self.CUR_ID, tf)

    def run(self):
        try:
            while self.CUR_ID < STOP_LIMIT:
                root, dirs, files = next(self.iterator)
                if dirs:
                    raise Exception("found an extra dir inside a subdir")
                for f in files:
                    self.q.append(f)
                while self.q and self.CUR_ID < STOP_LIMIT:
                    f = self.q.popleft()
                    self._process_file(root, f)
                    self.CUR_ID += 1
                    if self.CUR_ID % MAX_PROCESS_SIZE == 0:
                        self.mapping.offload()
                        self.inverted_index.offload()
                        self.inverted_index = InvertedIndex(self.inverted_index.num + 1)
        except StopIteration:
            print("Stop iteration received.")
        finally:
            print('offloading.')
            self.mapping.offload()
            self.inverted_index.offload()


def main():
    worker = Worker()
    worker.run()

if __name__ == '__main__':
    main()
