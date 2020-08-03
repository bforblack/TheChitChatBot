from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.cluster.util import cosine_distance
import networkx as nx
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer


class NlpEngineStart:
    def __init__(self,data):
        self.clean(data)

    def clean(self,Data):
        self.cleanData = []
        Data=Data.split(". ")
        for d in Data:
            self.cleanData.append(d.replace("[^a-zA-Z]", " ").split(" "))
        self.generateSimilarity_matrix(self.cleanData)


    def generateSimilarity_matrix(self,cleanData_List):

        similarity_matrix = np.zeros((len(cleanData_List), len(cleanData_List)))
        #working only for english will soon support multi Langugaes

        stop_words=stopwords.words('english')

        for idx1 in range(len(cleanData_List)):
            for idx2 in range(len(cleanData_List)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = self.sentence_similarity(cleanData_List[idx1], cleanData_List[idx2],stop_words)


        sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
        #eigval, eigvector = np.linalg.eig(similarity_matrix)
        #dominant_eigval = np.abs(eigval).max()
        #print("eiganVal",eigval)
        #print("eigvector",eigvector)
        #print("dominat_eigval",dominant_eigval)
        #self.scores = np.where(eigval == dominant_eigval)
        self.scores = nx.pagerank(sentence_similarity_graph)
        print("score",self.scores)
        nx.draw(sentence_similarity_graph,with_labels = True)
        plt.savefig("G:/filename.png")


    def sentence_similarity(self,sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
        stemmer = PorterStemmer()
        sent1 = [stemmer.stem(w.lower())for w in sent1]
        sent2 = [stemmer.stem(w.lower()) for w in sent2]
        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)


    def summary(self):
        summarize_text = []
        ranked_sentence = sorted(((self.scores[i], s) for i, s in enumerate(self.cleanData)), reverse=True)
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)

        for i in range(int(len(self.cleanData) / 2)):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        #print("Summarize Text: \n", ". ".join(summarize_text))
        return "Summarize Text: ", ". ".join(summarize_text)


