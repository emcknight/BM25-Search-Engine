import os
import pickle
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

class Indexer:
    # DB File name set to ir.idx
    db_file = "ir.idx" 

    # Init the Indexer object
    def __init__(self):
        self.words = []  # List to store all words in all documents
        self.raw_ds = None  # Dataset storage before tokenization                         
        self.cleaned_ds = []  # Dataset storage after tokenization
        self.corpus_stats = {'avgdl': 0.0,  # Corpus stats to track needed values for BM25 caluclations
                             'doc_len': [],
                             'tf': [],
                             'df': {},
                             'idf': {},
                             'doc_len': [],
                             'corpus_size': 0}           
        self.stopwords = stopwords.words('english')  # Stopwords from nltk package
       
        # If pickle exists, load it, else run the index creation function below
        if os.path.exists(self.db_file):
            self.load()
        else:
            # Prescribed dataset loading
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']

            self.clean_text(self.raw_ds)  # Clean text function call
            self.words = list(set(self.words))  # Save words as a list
            self.raw_ds = list(enumerate(self.raw_ds))  # Saves the raw dataset as an enumerated list. Giving each document an id
            self.cleaned_ds = list(enumerate(self.cleaned_ds))  # Same as above
            self.get_corpus_stats()  # Calls the get corpus stats function
            self.save()  # Save the index as a pickle file.

    # Function to save the Index object to the current folder.
    def save(self):
       f = open(self.db_file, 'wb')
       pickle.dump(self.__dict__, f, 2)
       f.close()

    # Function to load the Index when it already exists
    def load(self):
       f = open(self.db_file, 'rb')
       tmp_dict = pickle.load(f)
       f.close

       self.__dict__.update(tmp_dict)

    # Clean text function
    def clean_text(self, lst_text):
        # Lematizer and tokenizer creation
        lematizer = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'\w+')

        # Loops through each doc, tokenizes the lowercase version, removes stop words, lemmatizes, and then appends to the cleaned ds
        for doc in lst_text:
          doc = tokenizer.tokenize(doc.lower())
          doc = [w for w in doc if not w in self.stopwords and not w in 'cnn']
          doc = [lematizer.lemmatize(w) for w in doc]
          self.cleaned_ds.append(doc)

        return

    # Get corpus stats function
    def get_corpus_stats(self):
        # Loops through each cleaned document, ignoring the id of each document.
        for _, doc in self.cleaned_ds:
            self.corpus_stats['corpus_size'] += 1 
            self.corpus_stats['doc_len'].append(len(doc))

            # Get the term frequency for each term in the document and append to the stats
            freqs = {}
            for term in doc:
                term_count = freqs.get(term, 0) + 1
                freqs[term] = term_count
            self.corpus_stats['tf'].append(freqs)

            # For each unique term in the document, calculate the document frequency
            for term, _ in freqs.items():
                df_count = self.corpus_stats['df'].get(term, 0) + 1
                self.corpus_stats['df'][term] = df_count
        
        # For each unique term in all of the documents, calculate idf
        for term, freq in self.corpus_stats['df'].items():
            self.corpus_stats['idf'][term] = math.log(1 + (self.corpus_stats['corpus_size'] - freq + 0.5) / (freq + 0.5)) 
        
        self.corpus_stats['avgdl'] = sum(self.corpus_stats['doc_len']) / self.corpus_stats['corpus_size']  # Calculate the average document length
        
        return self

# SearchAgent class using BM25 formula
class SearchAgent:
    # Initial global variables for object
    k1 = 1.5      
    b = 0.75  
    sa_file = 'sa.idx'

    # Init the object
    def __init__(self, indexer):
        # If the pickle already exists, load it, otherwise create it
        if os.path.exists(self.sa_file):
            self.load()
        else:
            # Set calcs to index calcs for easier usage
            self.i = indexer
            self.doc_len = self.i.corpus_stats['doc_len']
            self.idf = self.i.corpus_stats['idf']
            self.avgdl = self.i.corpus_stats['avgdl']
            self.corpus_size = self.i.corpus_stats['corpus_size']
            self.tf = self.i.corpus_stats['tf']
            self.save()
        
    # Function to save the SearchAgent object to the current folder.
    def save(self):
       f = open(self.sa_file, 'wb')
       pickle.dump(self.__dict__, f, 2)
       f.close()

    # Function to load the SearchAgent when it already exists
    def load(self):
       f = open(self.sa_file, 'rb')
       tmp_dict = pickle.load(f)
       f.close

       self.__dict__.update(tmp_dict)

    # Function to get the score for a given document given a query
    def get_score(self, query, doc):
        score = 0.0

        # Get variables for the given doc
        doc_len = self.doc_len[doc]
        freqs = self.tf[doc]

        # For each term in the query..
        for term in query:
            # Do nothing if the term doesn't appear at all
            if term not in freqs:
                continue
            # Otherwise, calculate the BM25 score
            freq = freqs[term]
            score += ((self.idf[term] * freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        
        # Return the document and the score
        return doc, score
    
    # Function to search the corpus given a query
    def search(self, query):
        # Gets the score for each document given the query and sorts in descending order from highest score to lowest score
        scores = [self.get_score(query, doc) for doc in range(self.corpus_size)]  
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # Function called by the object to start the query process. Also contains text cleaning process
    def query(self, query):
        # Lematizer and tokenizer initialization
        lematizer = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'\w+')

        # Tokenize the query string and clean the text
        query = tokenizer.tokenize(query.lower())
        query = [w for w in query if not w in stopwords.words('english')]
        query = [lematizer.lemmatize(w) for w in query]

        results = self.search(query)  # Run the BM25 search on the corpus

        # Display the results if any exist
        if len(results) == 0:
            return None
        else:
            self.display_results(results)
    
    # Results display function. Shows the top 5 documents and the first 150 characters of the document
    def display_results(self, results):
        for docid, score in results[:5]: 
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid][1][slice(150)]+"...")

# Driver function that instantiates the index, the search agent, and then takes a query string from the user.
if __name__ == "__main__":
    i = Indexer()
    bm25 = SearchAgent(i)
    query = str(input('Please type a search string: '))
    bm25.query(query)
