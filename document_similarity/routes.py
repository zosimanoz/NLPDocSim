import os

import jsonpickle as jsonpickle
import nltk
from flask import Blueprint, render_template, request, url_for, redirect, abort, current_app, send_from_directory, \
    Response, jsonify
from gensim import corpora
from nltk import sent_tokenize, word_tokenize
import gensim
import numpy as np

nltk.download('punkt')

doc_sim = Blueprint('doc_sim', __name__,
                    template_folder='templates',
                    static_folder='static')


@doc_sim.route('/')
def doc_sim_index():
    return render_template('document_similarity/index.html')


@doc_sim.route('/', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    print(filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        print(file_ext)
        if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(current_app.config['UPLOAD_PATH'], filename))
    return render_template('document_similarity/index.html',
                           data=send_from_directory(current_app.config['UPLOAD_PATH'], filename))


@doc_sim.route('/process', methods=['POST'])
def process_form():
    data = request.get_json()
    text_one = data['text_one']
    text_two = data['text_two']
    query = data['test_query']

    # FOR DOC1
    file1_docs = []
    tokens = sent_tokenize(text_one)
    for line in tokens:
        file1_docs.append(line)

    len_doc_1 = len(file1_docs)

    gen_words_doc1 = [[w.lower() for w in word_tokenize(text)]
                      for text in file1_docs]

    dictionary = gensim.corpora.Dictionary(gen_words_doc1)
    # print(dictionary.token2id)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_words_doc1]
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity(current_app.config['WORK_DIR_PATH'], tf_idf[corpus],
                                          num_features=len(dictionary))

    # FOR DOC2
    file2_docs = []
    tokens = sent_tokenize(text_two)
    for line in tokens:
        file2_docs.append(line)

    len_doc_2 = len(file2_docs)
    gen_words_doc2 = [[w.lower() for w in word_tokenize(text)]
                      for text in file2_docs]

    dictionary2 = gensim.corpora.Dictionary(gen_words_doc2)
    corpus2 = [dictionary2.doc2bow(gen_doc) for gen_doc in gen_words_doc2]
    tf_idf2 = gensim.models.TfidfModel(corpus2)
    sims2 = gensim.similarities.Similarity(current_app.config['WORK_DIR_PATH'], tf_idf2[corpus2],
                                           num_features=len(dictionary2))

    # QUERY
    avg_sims = []
    avg_sim_doc_2 = []
    query_docs = []
    tokens = sent_tokenize(query)
    for line in tokens:
        query_docs.append(line)

    for line in query_docs:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow = dictionary.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf[query_doc_bow]
        print('Comparing Result With DOC 1:', sims[query_doc_tf_idf])
        sum_of_sims = (np.sum(sims[query_doc_tf_idf], dtype=np.float32))
        avg = sum_of_sims / len(file1_docs)
        print(f'avg: {sum_of_sims / len(file1_docs)}')
        avg_sims.append(avg)

    total_average_for_doc1 = np.sum(avg_sims, dtype=np.float)

    for line in query_docs:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow2 = dictionary2.doc2bow(query_doc)
        query_doc_tf_idf2 = tf_idf2[query_doc_bow2]
        print('Comparing Result With DOC 2:', sims2[query_doc_tf_idf2])
        sum_of_sims2 = (np.sum(sims2[query_doc_tf_idf2], dtype=np.float32))
        avg2 = sum_of_sims2 / len(file2_docs)
        print(f'avg2: {sum_of_sims2 / len(file2_docs)}')
        avg_sim_doc_2.append(avg2)

    total_average_for_doc2 = np.sum(avg_sim_doc_2, dtype=np.float)

    json_response = jsonify({ 'similarity_doc1':total_average_for_doc1, 'similarity_doc2': total_average_for_doc2})

    return json_response, 200
