from collections import defaultdict

# runtime: O(n) since runs through each character once
def tokenize_text(text: str) -> tuple[defaultdict, int]:
    """Tokenize text content directly and count token frequencies.

    Args:
        text: The text to tokenize

    Returns:
        defaultdict(int) with token frequencies (excluding stopwords if provided)
    """
    text = text.lower()
    cur = 0
    index = 0
    token_freq = defaultdict(int)
    total_count = 0
    while index < len(text):
        ch = text[index]
        if not (('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9')):
            token = text[cur:index]
            if token:
                token_freq[token] += 1
                total_count += 1
            cur = index + 1
        index += 1

    token = text[cur:index]
    if token:
        token_freq[token] += 1
        total_count += 1
    return token_freq, total_count

if __name__ == '__main__':
    print(tokenize_text('the word they have'))