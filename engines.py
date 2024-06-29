import math
import re
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from icecream import ic # For debugging

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
        elif self.use_stemming:
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


    

    # Add more scoring methods here as needed
    def calculate_scores(self, query_terms, metric='sum'):
        scores = {}
        for command, tf_idf_vector in self.tf_idf_matrix.items():
            if metric == 'sum':
                score = self._score_sum(query_terms, tf_idf_vector)
            # Add more metrics here and their corresponding method calls
            scores[command] = score

        return scores

    def recommend_command(self, query, top_n=1, metric='sum'):
        preprocessed_query = self.preprocess_query(query)
        query_terms = preprocessed_query

        scores = self.calculate_scores(query_terms, metric)

        # Sorting and returning the top n results in a pretty format
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results = [(cmd, scr, self.commands_set[cmd]) for cmd, scr in sorted_scores]
        return pd.DataFrame(results, columns=['Command', 'Score', 'Description']).to_string(index=False)

if __name__ == "__main__":
    dataset_path = './commands_dataset.txt'
    
    def load_dataset(file_path):
        commands_set = {}
        with open(file_path, 'r') as file:
            for line in file:
                command, description = line.strip().split(':',1)
                commands_set[command] = description
        return commands_set
    
    commands_set = load_dataset(dataset_path)
    tfidf_engine = TFIDFEngine(commands_set, use_lemmatization=False, use_stemming=False)
    tfidf_engine.fit()
    
    #query = input("Enter your command query: ")
    query = "Update generated configuration files"
    print(tfidf_engine.recommend_command(query, top_n=5))
