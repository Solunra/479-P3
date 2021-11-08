import test_resources
import nltk
import re
import datetime


def main():
    nltk.download('punkt', quiet=True)
    allDocuments = {}

    naive_time = 0
    spimi_time = 0
    # limited from 0 to 9 in order to get 10 000 docs
    for index in range(0, 10):
        document = test_resources.main(f'{str(index).zfill(3)}')
        allDocuments.update(document)
        # Uncomment for time
        start = datetime.datetime.now()
        naive_indexer(document)
        end = datetime.datetime.now()
        naive_time = naive_time + (end.timestamp() - start.timestamp())

        start = datetime.datetime.now()
        naive_indexer_spimi(document)
        end = datetime.datetime.now()
        spimi_time = spimi_time + (end.timestamp() - start.timestamp())

    with open(f'time_details.txt', 'w+') as file:
        file.write(f'naive took a total of {naive_time} ms\n')
        file.write(f'spimi took a total of {spimi_time} ms')

    global spimi_dictionary
    processed_vocabulary = lossy_compression_table(spimi_dictionary, get_stop_words_25(), get_stop_words_126())


naive_indexer_dictionary = {}


# Task 1; Accepts 'Documents' as a docId -> list of tokens
def naive_indexer(documents):
    global naive_indexer_dictionary

    # save as word -> [listOf]
    F_list = []
    # add to list
    for docId, document in documents.items():
        for token in document:
            F_list.append((docId, token))

    filtered_F_list = []
    processed_tokens = []
    for id, token in F_list:
        if token not in processed_tokens:
            filtered_F_list.append((id, token))
            processed_tokens.append(token)

    # sort into postings list
    for id, token in filtered_F_list:
        if token not in naive_indexer_dictionary.keys():
            naive_indexer_dictionary[token] = [id]
        else:
            naive_indexer_dictionary[token].append(id)


spimi_dictionary = {}


# Task 1; Accepts 'Documents' as a docId -> list of tokens
def naive_indexer_spimi(documents):
    # save as word -> [listOf]
    global spimi_dictionary
    # add to dictionary
    for docId, document in documents.items():
        for token in document:
            if token not in spimi_dictionary.keys():
                spimi_dictionary[token] = [docId]
            else:
                if docId not in spimi_dictionary[token]:
                    spimi_dictionary[token].append(docId)

    # sort dictionary values
    for key, value in spimi_dictionary.items():
        value.sort()
        spimi_dictionary[key] = value


# Task 2; Assume query is a singular term, vocabulary as the same as Task 1's return
def single_term_query_processing(query, vocabulary):
    doc_ids = vocabulary.get(query)
    if doc_ids is None:
        return []
    else:
        return doc_ids


# Task 3; Lossy Compression; documents is a list of tokens
def lossy_compression_table(vocabulary, stop_words_small, stop_words_big):
    # lower cases everything
    no_numbers_tokens = {token: vocabulary[token] for token in vocabulary if not re.search('\d', token)}
    case_folded_tokens = {}
    for token in no_numbers_tokens:
        lower_case_token = token.lower()
        if lower_case_token not in case_folded_tokens.keys():
            case_folded_tokens[lower_case_token] = no_numbers_tokens[token]
        else:
            current_tokens = case_folded_tokens[lower_case_token]
            other_tokens = no_numbers_tokens[token]
            if current_tokens is not None and other_tokens is not None:
                case_folded_tokens[lower_case_token] = list(set(current_tokens).union(set(other_tokens)))
    stop_word_filtered_tokens_25 = {token: case_folded_tokens[token] for token in case_folded_tokens if token not in stop_words_small}
    stop_word_filtered_tokens_126 = {token: stop_word_filtered_tokens_25[token] for token in stop_word_filtered_tokens_25 if token not in stop_words_big}
    ps = nltk.stem.PorterStemmer()
    stemmed_tokens = {}
    for token in stop_word_filtered_tokens_126:
        stemmed_token = ps.stem(token)
        if stemmed_token not in stemmed_tokens.keys():
            stemmed_tokens[stemmed_token] = stop_word_filtered_tokens_126[token]
        else:
            current_tokens = stemmed_tokens[stemmed_token]
            other_tokens = stop_word_filtered_tokens_126[token]
            if current_tokens is not None and other_tokens is not None:
                stemmed_tokens[stemmed_token] = list(set(current_tokens).union(set(other_tokens)))
    return stemmed_tokens


def get_stop_words_126():
    return ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
            'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
            'by', 'cannot', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few',
            'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers',
            'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
            'itself', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on',
            'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
            'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs',
            'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
            'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
            'which', 'while', 'who', 'whom', 'why', 'with', 'will', 'would', 'you', 'your', 'yours', 'yourself',
            'yourselves']


# From https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html#sec:stopwords
def get_stop_words_25():
    return ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'were', 'will', 'with']


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


