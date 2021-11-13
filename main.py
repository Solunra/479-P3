import math
import test_resources
import nltk
import datetime


# all global variables
average_document_length = 0
count_for_naive = 0
count_for_spimi = 0
naive_indexer_dictionary = {}
spimi_dictionary = {}
count_of_documents = 0
document_frequency = {}


def main():
    nltk.download('punkt', quiet=True)

    # Generates timing for naive and spimi
    print(f'Starting Subproject 1')
    subproject_1()

    print(f'Starting Subproject 2')
    subproject_2()

    print(f'Starting Subproject 2_term_comparison')
    subproject_2_single_term_comparison()


def subproject_2():
    global spimi_dictionary
    spimi_dictionary = {}
    global naive_indexer_dictionary
    naive_indexer_dictionary = {}

    allDocuments = {}
    for index in range(0, 22):
        document = test_resources.main(f'{str(index).zfill(3)}')
        allDocuments.update(document)
        naive_indexer_spimi(document)
    # Resets and gets the whole corpus into the dictionary
    get_document_frequency(allDocuments)
    generate_document_frequency(allDocuments)
    # Similar to SubProject 1, both are done with no compression techniques
    queries = ['Democrats’ welfare and healthcare reform policies', 'Drug company bankruptcies', 'George Bush']
    for index, query in enumerate(queries):
        get_ranked_list(allDocuments, spimi_dictionary, query, index)
    # Test Queries
    # BM25
    bm25_query = 'Shultz soybeans sales'
    b_values = [0.75, 0.5, 0.25]
    k1_values = [1.25, 1.5, 2]
    count = 0
    for b in b_values:
        for k1 in k1_values:
            get_ranked_list(allDocuments, spimi_dictionary, bm25_query, f'test-{count}', k_one=k1, b=b)
            count = count + 1

    # Multiple Keywords (AND)
    keyword_query = 'major mortgage lenders declined'
    and_result = query_processing_with_boolean_logic(spimi_dictionary, keyword_query, use_and=True)
    with open(f'./results/logical_and_results.txt', 'w+') as file:
        file.write(', '.join(and_result))
    # Multiple Keywords (OR)
    or_result = query_processing_with_boolean_logic(spimi_dictionary, keyword_query, use_and=False)
    with open(f'./results/logical_or_results.txt', 'w+') as file:
        file.write(', '.join(or_result))


def subproject_2_single_term_comparison():
    global spimi_dictionary
    spimi_dictionary = {}
    global naive_indexer_dictionary
    naive_indexer_dictionary = {}

    for index in range(0, 5):
        document = test_resources.main(f'{str(index).zfill(3)}')
        naive_indexer(document)
        naive_indexer_spimi(document)
    # Test Queries
    # Single term
    single_term = 'President'
    spimi_single_result = single_term_query_processing(single_term, spimi_dictionary)
    naive_single_result = single_term_query_processing(single_term, naive_indexer_dictionary)
    with open(f'./results/single_term_results.txt', 'w+') as file:
        file.write('SPIMI-Based: \n')
        file.write(', '.join(spimi_single_result))
        file.write('\n\nNaive-Based:\n')
        file.write(', '.join(naive_single_result))


def subproject_1():
    global spimi_dictionary
    spimi_dictionary = {}
    global naive_indexer_dictionary
    naive_indexer_dictionary = {}

    naive_time = 0
    spimi_time = 0
    for index in range(0, 22):
        document = test_resources.main(f'{str(index).zfill(3)}')
        # Gets time value for both indexing
        start = datetime.datetime.now()
        naive_indexer(document, timed=True)
        end = datetime.datetime.now()
        naive_time = naive_time + (end.timestamp() - start.timestamp())

        start = datetime.datetime.now()
        naive_indexer_spimi(document, timed=True)
        end = datetime.datetime.now()
        spimi_time = spimi_time + (end.timestamp() - start.timestamp())

    with open(f'./results/time_details.txt', 'w+') as file:
        file.write(f'naive took a total of {naive_time} s\n')
        file.write(f'spimi took a total of {spimi_time} s')


# processes boolean logic here
def query_processing_with_boolean_logic(dictionary, query, use_and=True):
    list_of_query_terms = query.split(' ')
    list_of_doc_ids = []
    for query_term in list_of_query_terms:
        list_of_doc_ids.append(dictionary[query_term])
    results = set(list_of_doc_ids.pop())
    for doc_ids in list_of_doc_ids:
        if use_and:
            results = results.intersection(doc_ids)
        else:
            results = results.union(doc_ids)
    sorted_results = list(results)
    sorted_results.sort(key=int)
    return sorted_results


def get_ranked_list(allDocuments, spimi_dictionary, query, index, k_one=1.2, b=0.75):
    global document_frequency
    score = bm25_query_processing(query, allDocuments, spimi_dictionary, document_frequency, k_one=k_one, b=b)
    # sorting from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    sorted_scores = {key: value for key, value in sorted(score.items(), key=lambda item: item[1], reverse=True)}
    with open(f'./results/scores-{index}.txt', 'w+') as file:
        file.write(f'For query: {query} with k1={k_one} and b={b}\n')
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


# Accepts 'Documents' as a docId -> list of tokens
def naive_indexer(documents, timed=False):
    global naive_indexer_dictionary
    global count_for_naive

    if count_for_naive == 10000 and timed:
        return

    # save as word -> [listOf]
    F_list = []
    # add to list
    for docId, document in documents.items():
        for token in document:
            if (docId, token) not in F_list:
                F_list.append((docId, token))

    # taken and adapted from https://stackoverflow.com/questions/59687010/python-sort-list-consisting-of-int-string-pairs-descending-by-int-and-ascend
    F_list = [(docId, token) for docId, token in sorted(F_list, key=lambda x: x[1])]

    # add into postings list
    for id, token in F_list:
        if count_for_naive == 10000 and timed:
            return
        if token not in naive_indexer_dictionary.keys():
            naive_indexer_dictionary[token] = [id]
            count_for_naive = count_for_naive + 1
        else:
            if id not in naive_indexer_dictionary[token]:
                naive_indexer_dictionary[token].append(id)
                count_for_naive = count_for_naive + 1


# Accepts 'Documents' as a docId -> list of tokens
def naive_indexer_spimi(documents, timed=False):
    # save as word -> [listOf]
    global spimi_dictionary
    global count_of_documents
    global count_for_spimi

    if count_for_spimi == 10000 and timed:
        return

    # add to dictionary
    for docId, document in documents.items():
        count_of_documents = count_of_documents + 1
        for token in document:
            if len(spimi_dictionary) == 10000 and timed:
                return
            if token not in spimi_dictionary.keys():
                spimi_dictionary[token] = [docId]
                count_for_spimi = count_for_spimi + 1
            else:
                if docId not in spimi_dictionary[token]:
                    spimi_dictionary[token].append(docId)
                    count_for_spimi = count_for_spimi + 1


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


