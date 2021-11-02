import re
import nltk


# All code taken from P1
def main(file_number):
    # quiet so it doesn't output to terminal about it being downloaded already
    nltk.download('punkt', quiet=True)
    return run_all_in_pipeline(f'./reuters/reut2-{file_number}.sgm')


def run_all_in_pipeline(file_location):
    dict_of_texts = read_and_extract_raw_text(file_location)
    dict_of_tokens = tokenize(dict_of_texts)
    return dict_of_tokens


def read_and_extract_raw_text(file_location):
    unallowed_symbols = ['&', '<', '>']
    all_text = ""
    all_body_and_title = ""
    with open(file_location) as file:
        all_text += file.read()
    split_text = all_text.split('</REUTERS>')
    # returned dict of id -> list of tokens
    dict_of_texts = {}
    for text in split_text:
        if text:
            all_body_and_title = ""
            # Using regex to find the newID of the reuters collection
            doc_id = re.findall(f'NEWID="(.*)">', text, flags=re.DOTALL)
            # Using REGEX to find all files between the specified tags
            # Both TITLE and BODY only occurs at specific points
            title = re.findall(f"<TITLE>(.*)</TITLE>", text, flags=re.DOTALL)
            body = re.findall(f"<BODY>(.*)</BODY>", text, flags=re.DOTALL)
            if title:
                all_body_and_title += title[0] + ' '
            if body:
                all_body_and_title += body[0] + ' '
            dict_of_texts[doc_id] = all_body_and_title
    return dict_of_texts


def string_contains_list(str_to_check, unallowed_chars):
    for element in unallowed_chars:
        result = str_to_check.find(element)
        if result != -1:
            return True
    return False


def tokenize(dict_of_texts):
    dict_of_tokens = {}
    for docID, text in dict_of_texts.items():
        tokens = nltk.word_tokenize(text)
        dict_of_tokens[docID] = tokens
    return dict_of_tokens