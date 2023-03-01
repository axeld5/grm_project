import spacy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


def preprocess_review(review, label, nlp, graph_visu=False):

    ## preprocess of sentences to have a graph that "sees" sentences     
    
    sentences = create_sentence_list(review, nlp)
    dico_words = tokenize_words(sentences)
    G = create_graph(sentences, dico_words)
    if graph_visu:
        visualise_graph(G)
    node_features, edges, label = define_graph_features(G, nlp, label)

    ## Create a PyTorch Geometric Data object
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    y = torch.tensor(label, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def create_sentence_list(review, nlp):
    doc = nlp(review)
    sentences = [sent for sent in doc.sents]
    sentences = [[token.text for token in sent] for sent in sentences]
    sentences = [[word for word in sent if word.isalpha()] for sent in sentences]
    sentences = [[word for word in sent if not nlp.vocab[word].is_stop] for sent in sentences]
    sentences = [sent for sent in sentences if len(sent) > 0]
    return sentences

def tokenize_words(sentences):
    dico_words = {}
    for i,sent in enumerate(sentences):
        for word in sent:
            if word not in dico_words:
                dico_words[word] = [i]
            else:
                dico_words[word] += [i]
    return dico_words

def create_graph(sentences, dico_words):
    G = nx.Graph()
    G.add_nodes_from(dico_words.keys())
    ## Linking sentences in the graph 
    for i in range(len(sentences)-1):
        previous_word = sentences[i][-1]
        next_word = sentences[i+1][0]
        if previous_word != next_word:
            G.add_edge(previous_word, next_word, weight = 1)

    for word in dico_words.keys():
        for word2 in dico_words.keys():
            if word != word2:
                common_sentences = set(dico_words[word]).intersection(set(dico_words[word2]))
                if len(common_sentences) > 0:
                    G.add_edge(word, word2, weight = 1+len(common_sentences))
    return G

def define_graph_features(G, nlp, label):
    # For the GNN : 
    # nodes 
    node_features = []
    for node in G.nodes():
        node_features.append(nlp.vocab[node].vector)
    node_features = np.array(node_features)
    # edges
    edges = []
    for edge in G.edges():
        edges.append([list(G.nodes()).index(edge[0]), list(G.nodes()).index(edge[1])])
    edges = np.array(edges)
    # label
    label = [int(label)]

    return node_features, edges, label

def visualise_graph(G):
    plt.figure(figsize=(10,10))
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()