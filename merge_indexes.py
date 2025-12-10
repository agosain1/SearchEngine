import os
import json
import varint
import math
import struct
import bbhash
import hashlib

ROOT_PATH = '/Users/arnavgosain/Documents/cs121/'

class Posting:
    def __init__(self, short_id, score):
        self.short_id = short_id
        self.score = score

def find_index_files():
    index_files = []
    i = 1
    while os.path.exists(ROOT_PATH + f'inverted_index{i}.bin'):
        index_files.append(i)
        i += 1
    return index_files

def read_index_file(num):
    bin_path = ROOT_PATH + f'inverted_index{num}.bin'
    lex_path = ROOT_PATH + f'lexicon{num}.json'

    with open(lex_path, 'r') as f:
        lexicon = json.load(f)

    with open(bin_path, 'rb') as f:
        for term in sorted(lexicon.keys()):
            offset = lexicon[term]
            f.seek(offset)

            # Read term length and term
            term_len = varint.decode_stream(f)
            term_bytes = f.read(term_len)
            term_read = term_bytes.decode('utf-8')

            # Read postings count
            postings_count = varint.decode_stream(f)

            # Decode postings
            postings = []
            last_id = 0
            for _ in range(postings_count):
                delta = varint.decode_stream(f)
                score = varint.decode_stream(f)

                doc_id = last_id + delta
                last_id = doc_id
                tf = score / 10000

                postings.append(Posting(doc_id, tf))

            yield term, postings

def merge_two_indexes(index1_num, index2_num, output_num):
    """Merge two index files into a new one, writing to disk incrementally"""
    print(f"Merging index {index1_num} and {index2_num} -> {output_num}...")

    out_path = ROOT_PATH + f'inverted_index{output_num}.bin'
    lex_path = ROOT_PATH + f'lexicon{output_num}.json'

    # Create generators for both indexes
    iter1 = read_index_file(index1_num)
    iter2 = read_index_file(index2_num)

    lexicon = {}
    offset = 0

    # Get first terms from both indexes
    try:
        term1, postings1 = next(iter1)
    except StopIteration:
        term1 = None
        postings1 = None

    try:
        term2, postings2 = next(iter2)
    except StopIteration:
        term2 = None
        postings2 = None

    with open(out_path, 'wb') as f:
        # Merge using two-pointer technique
        while term1 is not None or term2 is not None:
            # Determine which term to write
            if term1 is None:
                term_to_write = term2
                postings_to_write = postings2
                try:
                    term2, postings2 = next(iter2)
                except StopIteration:
                    term2 = None
                    postings2 = None
            elif term2 is None:
                term_to_write = term1
                postings_to_write = postings1
                try:
                    term1, postings1 = next(iter1)
                except StopIteration:
                    term1 = None
                    postings1 = None
            elif term1 < term2:
                term_to_write = term1
                postings_to_write = postings1
                try:
                    term1, postings1 = next(iter1)
                except StopIteration:
                    term1 = None
                    postings1 = None
            elif term2 < term1:
                term_to_write = term2
                postings_to_write = postings2
                try:
                    term2, postings2 = next(iter2)
                except StopIteration:
                    term2 = None
                    postings2 = None
            else:  # term1 == term2, merge postings
                term_to_write = term1
                postings_to_write = postings1 + postings2
                postings_to_write = sorted(postings_to_write, key=lambda p: p.short_id)
                try:
                    term1, postings1 = next(iter1)
                except StopIteration:
                    term1 = None
                    postings1 = None
                try:
                    term2, postings2 = next(iter2)
                except StopIteration:
                    term2 = None
                    postings2 = None

            term_offset = offset
            lexicon[term_to_write] = term_offset

            term_bytes = term_to_write.encode('utf-8')

            b = varint.encode(len(term_bytes))
            f.write(b)
            offset += len(b)

            f.write(term_bytes)
            offset += len(term_bytes)

            b = varint.encode(len(postings_to_write))
            f.write(b)
            offset += len(b)

            postings_sorted = sorted(postings_to_write, key=lambda p: p.short_id)
            last_id = 0
            for p in postings_sorted:
                delta = p.short_id - last_id
                last_id = p.short_id

                b = varint.encode(delta)
                f.write(b)
                offset += len(b)

                score_int = int(round(p.score * 10000))
                b = varint.encode(score_int)
                f.write(b)
                offset += len(b)

    with open(lex_path, 'w') as lf:
        json.dump(lexicon, lf)

    print(f"  Merged into index {output_num} with {len(lexicon)} terms")

def calculate_idf_from_all_indexes(index_numbers):
    """Calculate IDF scores from all index files"""
    print("Calculating IDF scores...")

    idf = {}
    all_docs = set()

    for num in index_numbers:
        for term, postings in read_index_file(num):
            for posting in postings:
                all_docs.add(posting.short_id)

            # Count unique docs for this term across all indexes
            if term not in idf:
                idf[term] = set()
            for posting in postings:
                idf[term].add(posting.short_id)

    total_docs = len(all_docs)
    print(f"Total documents: {total_docs}")

    # Convert doc sets to IDF scores (rounded to 4 decimal places)
    idf_scores = {}
    for term, doc_set in idf.items():
        df = len(doc_set)
        if df > 0:
            idf_scores[term] = round(math.log(total_docs / df), 4)

    idf_path = ROOT_PATH + 'idf_merged.json'
    with open(idf_path, 'w') as f:
        json.dump(idf_scores, f)

    print(f"Saved IDF scores to {idf_path}")
    return idf_scores

def replace_scores_with_idf(bin_path, lex_path, idf_scores):
    """Replace TF scores in the index with TF * IDF scores"""
    print("Replacing TF scores with TF * IDF scores...")

    # Read the current index
    with open(lex_path, 'r') as f:
        lexicon = json.load(f)

    # Create new binary file with TF * IDF scores
    temp_path = bin_path + '.tmp'
    with open(bin_path, 'rb') as f_in:
        with open(temp_path, 'wb') as f_out:
            new_lexicon = {}
            offset = 0

            for term in sorted(lexicon.keys()):
                old_offset = lexicon[term]
                f_in.seek(old_offset)

                # Read term length and term
                term_len = varint.decode_stream(f_in)
                term_bytes = f_in.read(term_len)
                term_read = term_bytes.decode('utf-8')

                # Read postings count
                postings_count = varint.decode_stream(f_in)

                # Read postings with TF scores
                postings = []
                last_id = 0
                for _ in range(postings_count):
                    delta = varint.decode_stream(f_in)
                    tf_int = varint.decode_stream(f_in)
                    doc_id = last_id + delta
                    last_id = doc_id
                    tf = tf_int / 10000
                    postings.append((doc_id, tf))

                # Write term offset to new lexicon
                new_offset = offset
                new_lexicon[term] = new_offset

                # Write term length and term
                b = varint.encode(term_len)
                f_out.write(b)
                offset += len(b)

                f_out.write(term_bytes)
                offset += len(term_bytes)

                # Write postings count
                b = varint.encode(postings_count)
                f_out.write(b)
                offset += len(b)

                # Get IDF score for this term (default to 0 if not found)
                idf = idf_scores.get(term, 0)

                # Write postings with TF * IDF
                last_id = 0
                for doc_id, tf in postings:
                    delta = doc_id - last_id
                    last_id = doc_id

                    b = varint.encode(delta)
                    f_out.write(b)
                    offset += len(b)

                    # Calculate TF * IDF and convert to int
                    tf_idf = tf * idf
                    tf_idf_int = int(round(tf_idf * 10000))
                    b = varint.encode(tf_idf_int)
                    f_out.write(b)
                    offset += len(b)

    # Replace original file with new one
    os.remove(bin_path)
    os.rename(temp_path, bin_path)

    # Update lexicon
    with open(lex_path, 'w') as f:
        json.dump(new_lexicon, f)

    print("Replaced TF scores with TF * IDF scores")
    return new_lexicon

def deterministic_hash(s):
    """Deterministic hash function using SHA256"""
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

def create_mphf_lexicon(json_lexicon_path, mphf_path, offsets_path):
    print("Loading JSON lexicon...")
    with open(json_lexicon_path, "r") as f:
        lexicon = json.load(f)

    terms = sorted(lexicon.keys())

    print("Hashing keys...")
    # convert to 64-bit unsigned ints using deterministic hash
    hashed_terms = [deterministic_hash(t) & 0xFFFFFFFFFFFFFFFF for t in terms]

    print("Building MPHF...")
    # PyMPHF constructor: keys_list, number_of_keys, gamma, seed
    mph = bbhash.PyMPHF(hashed_terms, len(hashed_terms), 1.0, 0)

    print("Saving MPHF...")
    mph.save(mphf_path)

    print("Saving offsets...")
    with open(offsets_path, "wb") as f:
        for t in terms:
            f.write(struct.pack("<Q", lexicon[t]))

    print("Done.")
    return mph

def merge_all_indexes():
    """Main merge function - merges indexes 2 at a time"""
    index_numbers = find_index_files()

    if not index_numbers:
        print("No index files found!")
        return

    if len(index_numbers) == 1:
        print("Only one index file found, no merge needed!")
        return

    print(f"Found {len(index_numbers)} index files: {index_numbers}")

    # repeatedly merge pairs until only one remains
    current_indexes = index_numbers[:]
    next_merge_num = max(current_indexes) + 1

    while len(current_indexes) > 1:
        next_round = []

        i = 0
        while i < len(current_indexes):
            if i + 1 < len(current_indexes):
                # Merge pair
                idx1 = current_indexes[i]
                idx2 = current_indexes[i + 1]
                merge_two_indexes(idx1, idx2, next_merge_num)
                next_round.append(next_merge_num)
                next_merge_num += 1
                i += 2
            else:
                # Odd one out, keep for next round
                next_round.append(current_indexes[i])
                i += 1

        current_indexes = next_round
        print(f"Round complete: {len(current_indexes)} indexes remaining")

    final_index = current_indexes[0]
    print(f"\nFinal merged index: inverted_index{final_index}.bin")

    final_bin = ROOT_PATH + f'inverted_index{final_index}.bin'
    final_lex = ROOT_PATH + f'lexicon{final_index}.json'
    merged_bin = ROOT_PATH + 'inverted_index_merged.bin'

    idf_scores = calculate_idf_from_all_indexes(index_numbers)

    replace_scores_with_idf(final_bin, final_lex, idf_scores)

    # Rename final binary to merged binary
    os.rename(final_bin, merged_bin)

    # Create MPHF lexicon and offsets instead of storing JSON
    mphf_path = ROOT_PATH + 'lexicon_merged.mphf'
    offsets_path = ROOT_PATH + 'offsets.bin'
    create_mphf_lexicon(final_lex, mphf_path, offsets_path)

    # Delete intermediate index files
    print(f"\nCleaning up intermediate files...")
    for num in index_numbers:
        bin_file = ROOT_PATH + f'inverted_index{num}.bin'
        lex_file = ROOT_PATH + f'lexicon{num}.json'
        if os.path.exists(bin_file):
            os.remove(bin_file)
        if os.path.exists(lex_file):
            os.remove(lex_file)

    # Delete intermediate merged files
    for num in range(max(index_numbers) + 1, next_merge_num):
        bin_file = ROOT_PATH + f'inverted_index{num}.bin'
        lex_file = ROOT_PATH + f'lexicon{num}.json'
        if os.path.exists(bin_file):
            os.remove(bin_file)
        if os.path.exists(lex_file):
            os.remove(lex_file)

    # Delete temporary JSON lexicon and IDF file
    if os.path.exists(final_lex):
        os.remove(final_lex)
    idf_path = ROOT_PATH + 'idf_merged.json'
    if os.path.exists(idf_path):
        os.remove(idf_path)

    print(f"\nMerge complete!")
    print(f"Final output files:")
    print(f"  - {merged_bin}")
    print(f"  - {mphf_path}")
    print(f"  - {offsets_path}")

if __name__ == '__main__':
    merge_all_indexes()
