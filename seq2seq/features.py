import math
import numpy as np
import operator


class Document:

    def __init__(self, docid, q, ans):
        self.docid = docid
        self.q = q
        self.ans = ans
        self.q_words = [x.strip() for x in self.q.split(" ")]
        self.ans_words = [x.strip() for x in self.ans.split(" ")]
        self.words = self.q_words + self.ans_words
        self.bow = {}
        self.set_bow()

    def set_bow(self):
        for w in self.words:
            if w not in self.bow:
                self.bow[w] = 1
            else:
                self.bow[w] += 1


class Features:

    def __init__(self, documents):
        self.documents = documents
        self.doc_size = len(documents)
        self.word_size = 0
        self.total_word_count = 0
        self.word_index_dict = {}
        self.word_count_dict = {}

        # for tf-idf, # of docs where the term t appears
        self.term_doc_dict = {}

        # for chi square
        # category_dict['health'] = total # of terms in 'health' category
        self.category_dict = {}
        # term_dict['water'] = total # of 'water' in all documents
        self.term_dict = {}
        # category_term_dict['health']['water'] = total # of word 'water' in 'health' category
        self.category_term_dict = {}
        self.word_counting()

    def trim_sentence(self, sentence):
        return sentence
        #return self.komoran.nouns(sentence)

    def word_counting(self):
        for doc in self.documents:
            for w in doc.words:
                self.total_word_count += 1

                if w not in self.term_doc_dict:
                    self.term_doc_dict[w] = {}
                self.term_doc_dict[w][doc.docid] = 1

                if w not in self.word_count_dict:
                    self.word_count_dict[w] = 1
                    self.word_index_dict[w] = self.word_size
                    self.word_size += 1
                else:
                    self.word_count_dict[w] += 1

