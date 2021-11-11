import math
import test_resources
import nltk
import re
import datetime


average_document_length = 0


def main():
    nltk.download('punkt', quiet=True)
    allDocuments = {}

    naive_time = 0
    spimi_time = 0
    # limited from 0 to 9 in order to get 10 000 docs
    for index in range(0, 10):
        document = test_resources.main(f'{str(index).zfill(3)}')
        allDocuments.update(document)
        # Uncomment for time value
        # start = datetime.datetime.now()
        # naive_indexer(document, timed=True)
        # end = datetime.datetime.now()
        # naive_time = naive_time + (end.timestamp() - start.timestamp())
        #
        # start = datetime.datetime.now()
        # naive_indexer_spimi(document, timed=True)
        # end = datetime.datetime.now()
        # spimi_time = spimi_time + (end.timestamp() - start.timestamp())

    get_document_frequency(allDocuments)
    generate_document_frequency(allDocuments)
    global spimi_dictionary
    spimi_dictionary = {}
    naive_indexer_spimi(allDocuments)

    # with open(f'./results/time_details.txt', 'w+') as file:
    #     file.write(f'naive took a total of {naive_time} ms\n')
    #     file.write(f'spimi took a total of {spimi_time} ms')

    # Similar to SubProject 1, both are done with no compression techniques
    queries = ['Democrats’ welfare and healthcare reform policies', 'Drug company bankruptcies', 'George Bush']
    for index, query in enumerate(queries):
        get_ranked_list(allDocuments, spimi_dictionary, query, index)


def get_ranked_list(allDocuments, spimi_dictionary, query, index):
    global document_frequency
    score = bm25_query_processing(query, allDocuments, spimi_dictionary, document_frequency)
    # sorting from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    sorted_scores = {key: value for key, value in sorted(score.items(), key=lambda item: item[1], reverse=True)}
    with open(f'./results/scores-{index}.txt', 'w+') as file:
        file.write(f'For query: {query}\n')
        file.write(f'[ \n')
        for doc_id, doc_score in sorted_scores.items():
            file.write(f'{doc_id}, \n')
        file.write(f' ]')


def get_document_frequency(all_documents):
    global average_document_length
    total = 0
    count = 0
    for _, tokens in all_documents.items():
        total = total + len(tokens)
        count = count + 1
    average_document_length = total / count


# equations taken from https://nlp.stanford.edu/IR-book/pdf/11prob.pdf 11.32's equation
def bm25_query_processing(query, all_documents, dictionary, doc_frequency, k_one=1.2, b=0.75):
    global average_document_length
    list_of_terms = query.split(' ')
    score = {}
    for term in list_of_terms:
        if dictionary.get(term, None) is not None:
            for docId in dictionary[term]:
                # calculate idf here
                term_idf = math.log((count_of_documents/len(dictionary[term])))
                bm_numerator = doc_frequency[term][docId] * (k_one + 1)
                bm_denominator = doc_frequency[term][docId] + k_one * ((1 - b) + (b * (len(all_documents[docId]) / average_document_length)))
                if docId not in score.keys():
                    score[docId] = term_idf * bm_numerator / bm_denominator
                else:
                    score[docId] = score[docId] + (term_idf * bm_numerator / bm_denominator)
    return score


naive_indexer_dictionary = {}


# Task 1; Accepts 'Documents' as a docId -> list of tokens
def naive_indexer(documents, timed=False):
    global naive_indexer_dictionary
    count = 0
    if count == 10000 and timed:
        return

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
            count = count + 1
        else:
            if id not in naive_indexer_dictionary[token]:
                naive_indexer_dictionary[token].append(id)
                count = count + 1


spimi_dictionary = {}
count_of_documents = 0


# Task 1; Accepts 'Documents' as a docId -> list of tokens
def naive_indexer_spimi(documents, timed=False):
    # save as word -> [listOf]
    global spimi_dictionary
    global count_of_documents
    count = 0
    if count == 10000 and timed:
        return

    # add to dictionary
    for docId, document in documents.items():
        count_of_documents = count_of_documents + 1
        for token in document:
            if len(spimi_dictionary) == 10000:
                return
            if token not in spimi_dictionary.keys():
                spimi_dictionary[token] = [docId]
                count = count + 1
            else:
                if docId not in spimi_dictionary[token]:
                    spimi_dictionary[token].append(docId)
                    count = count + 1

    # sort dictionary values
    for key, value in spimi_dictionary.items():
        value.sort()
        spimi_dictionary[key] = value


document_frequency = {}


# Creates a list of documents and term frequency using positions
def generate_document_frequency(documents):
    global document_frequency
    for docId, document in documents.items():
        for token in document:
            if token not in document_frequency.keys():
                document_frequency[token] = {docId: 1}
            else:
                if docId not in document_frequency[token]:
                    document_frequency[token][docId] = 1
                else:
                    document_frequency[token][docId] = document_frequency[token][docId] + 1


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


