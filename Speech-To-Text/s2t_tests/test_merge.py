import difflib

def merge_texts(text1, text2):
    s = difflib.SequenceMatcher(None, text1, text2)
    match = s.find_longest_match(0, len(text1), 0, len(text2))
    
    if match.size == 0:
        return text1 + text2

    overlap_start = match.b
    return text1 + text2[overlap_start + match.size:]

# Example usage
text1 = "Hello, this is an example of a text ch"
text2 = "ton a text chunk with significant overlap."
merged_text = merge_texts(text1, text2)
print(merged_text)