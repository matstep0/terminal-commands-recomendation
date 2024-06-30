import math
import re
import numpy as np
from abc import ABC, abstractmethod
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from icecream import ic # For debugging
from scipy.special import kl_div, rel_entr
from scipy.stats import pearsonr


# Suppressing the verbose output of nltk downloader
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass  # Handle the exception silently if nltk resources are not downloaded successfully.

class Engine(ABC):
    def __init__(self, commands_set, use_lemmatization=False, use_stemming=False):
        self.commands_set = commands_set
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.preprocessed_commands = self._preprocess_commands(commands_set)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def recommend_command(self, query, top_n=1, metric='sum'):
        pass

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def _preprocess_commands(self, commands_set):
        preprocessed_commands = {}
        for command, description in commands_set.items():
            tokens = self.preprocess_text(description)
            preprocessed_commands[command]=tokens
        return preprocessed_commands

class TFIDFEngine(Engine):
    def __init__(self, commands_set, use_lemmatization=False, use_stemming=False):
        super().__init__(commands_set, use_lemmatization, use_stemming)
        self.tf_idf_matrix = None
        self.idf = None

    def fit(self):
        # Calculate IDF values
        N = len(self.preprocessed_commands)
        df = Counter()
        for description in self.preprocessed_commands.values():
            terms = set(description)  # Unique terms per document
            for term in terms:
                df[term] += 1

        self.idf = {term: math.log(N / df_term) for term, df_term in df.items()}

        # Calculate TF-IDF matrix
        self.tf_idf_matrix = {}
        for command, description in self.preprocessed_commands.items():
            tf = Counter(description)
            tf_idf_vector = {term: (1 + math.log(freq)) * self.idf[term] for term, freq in tf.items()}
            self.tf_idf_matrix[command] = tf_idf_vector

    def preprocess_query(self, query):
        tokens = self.preprocess_text(query)
        return tokens

    def _score_sum(self, query_terms, tf_idf_vector):
        query_tf = Counter(query_terms)
        #ic(query_tf)
        intersection = set(query_terms).intersection(set(tf_idf_vector.keys()))
        #ic(intersection)
        return sum((1 + math.log(query_tf[term])) * tf_idf_vector[term] for term in intersection)
    
    def _cosine_similarity(self, query_terms, tf_idf_vector):
        query_tf = Counter(query_terms)
        query_tf_idf = {term: (1 + math.log(freq)) * self.idf.get(term, 0) for term, freq in query_tf.items()}

        dot_product = sum(query_tf_idf[term] * tf_idf_vector.get(term, 0) for term in query_tf_idf)
        query_norm = math.sqrt(sum(value ** 2 for value in query_tf_idf.values()))
        vector_norm = math.sqrt(sum(value ** 2 for value in tf_idf_vector.values()))

        if not query_norm or not vector_norm:
            return 0.0

        return dot_product / (query_norm * vector_norm)

    def _kl_divergence(self, query_terms, tf_idf_vector):
        query_tf = Counter(query_terms)
        query_tf_idf = {term: (1 + math.log(freq)) * self.idf.get(term, 0) for term, freq in query_tf.items()}

        # Ensure both vectors have the same keys
        all_terms = set(query_tf_idf.keys()).union(set(tf_idf_vector.keys()))
        p = [query_tf_idf.get(term, 0) for term in all_terms]
        q = [tf_idf_vector.get(term, 0) for term in all_terms]
        
        return -sum(rel_entr(p, q)) # Minus is needed here

    from scipy.stats import pearsonr

    def _score_pearson_intersection(self, query_terms, tf_idf_vector):
        #This is special modification of pearson correlation, which takes into account only the intersection of terms. (Faster but with less accuracy)

        query_tf = Counter(query_terms)
        query_tf_idf = {term: (1 + math.log(query_tf[term])) * self.idf.get(term, 0) for term in query_terms}
        
        # Intersection of terms
        common_terms = set(query_tf_idf.keys()).intersection(tf_idf_vector.keys())
        if len(common_terms) < 2:  # Pearson wymaga co najmniej dwóch wartości
            return 0.0
        
        query_vector = [query_tf_idf[term] for term in common_terms]
        doc_vector = [tf_idf_vector[term] for term in common_terms]
        
        return pearsonr(query_vector, doc_vector)[0]  # Zwraca współczynnik korelacji Pearsona


    def _pearson_correlation(self, query_terms, tf_idf_vector):
        query_tf = Counter(query_terms)
        query_tf_idf = {term: (1 + math.log(freq)) * self.idf.get(term, 0) for term, freq in query_tf.items()}

        all_terms = set(query_tf_idf.keys()).union(set(tf_idf_vector.keys()))
        p = [query_tf_idf.get(term, 0) for term in all_terms]
        q = [tf_idf_vector.get(term, 0) for term in all_terms]

        if len(p) < 2 or len(q) < 2:
            return 0.0

        correlation, _ = pearsonr(p, q)
        return correlation


    def _score_jsd(self, query_terms, tf_idf_vector, epsilon=1e-10):
        def jensen_shannon_divergence(P, Q):
            M = 0.5 * (P + Q)
            return 0.5 * (np.sum(rel_entr(P, M)) + np.sum(rel_entr(Q, M)))
        
        query_tf = Counter(query_terms)
        query_tf_idf = {term: (1 + math.log(query_tf[term])) * self.idf.get(term, 0) for term in query_terms}
        
        # Normalize the query TF-IDF vector to get probabilities
        query_total = sum(query_tf_idf.values())
        if query_total == 0:
            return float('inf')  # Return infinity if the query TF-IDF sum is zero
        query_prob = {term: tfidf / query_total for term, tfidf in query_tf_idf.items()}
        
        # Normalize the document TF-IDF vector to get probabilities
        doc_total = sum(tf_idf_vector.values())
        doc_prob = {term: tfidf / doc_total for term, tfidf in tf_idf_vector.items()}

        # Align the two probability vectors by their terms (union of terms)
        all_terms = set(query_prob.keys()).union(doc_prob.keys())
        query_prob_aligned = np.array([query_prob.get(term, 0) + epsilon for term in all_terms])
        doc_prob_aligned = np.array([doc_prob.get(term, 0) + epsilon for term in all_terms])

        # Compute Jensen-Shannon Divergence
        js_divergence = jensen_shannon_divergence(query_prob_aligned, doc_prob_aligned)
        return -js_divergence


    def calculate_scores(self, query_terms, metric='sum'):
        metric_functions = {
            'sum': self._score_sum,
            'cosine': self._cosine_similarity,
            'kld': self._kl_divergence,
            'pearson': self._pearson_correlation,
            'pearson_intersection': self._score_pearson_intersection,
            'jsd': self._score_jsd
        }

        scores = {}
        for command, tf_idf_vector in self.tf_idf_matrix.items():
            score = metric_functions[metric](query_terms, tf_idf_vector)
            scores[command] = score

        return scores

    def recommend_command(self, query, top_n=1, metric='sum'):
        preprocessed_query = self.preprocess_query(query)
        query_terms = preprocessed_query

        scores = self.calculate_scores(query_terms, metric)

        # Sorting and returning the top n results in a pretty format
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results = [(cmd, scr, self.commands_set[cmd]) for cmd, scr in sorted_scores]
        ret = [(cmd, scr, self.commands_set[cmd]) for cmd, scr in sorted_scores]
        return ret
    