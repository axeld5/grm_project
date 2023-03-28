import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data

#nlp is a spacy model
def biagram_preprocessing(review, label, nlp, method="directed_bi"):
    #methods : directed_bi, undirected_bi, weighted_bi
    weights_each_type = {'ADJ': 3, 'ADV': 2, 'NOUN': 1, 'VERB': 4, 'ADP': 1, 'DET': 1, 'NUM': 1, 'PUNCT': 1, 'PRON': 1, 'PROPN': 1, 'SCONJ': 1, 'SYM': 1, 'X': 1, 'PART': 1, 'CCONJ': 1, 'INTJ': 1, 'AUX': 1, 'SPACE': 1, '': 1}

    ## sentences preprocessing
    sentences = preprocess_sentences(nlp, review)

    ## get the biagrams
    biagrams = get_biagrams(sentences)
    ### concatenate all the sentences in one
    sent = {}
    for sentence in sentences:
        sent.update(sentence)
    ### dico of how many times a biagram appears in the review
    dico_biagrams = get_weighted_biagram_appearance(biagrams, sent, weights_each_type, method)
    list_of_words = [word for sent in sentences for word in sent]
    list_of_words = list(set(list_of_words))
    ## create graph 
    G = biagrams_to_nx_graph(list_of_words, dico_biagrams)
    node_features, edges, edges_attr = get_graph_features(G, nlp)  
    # Get the label
    label_value = int(label)
    # Create a PyTorch Geometric Data object
    data = features_to_torch(node_features, edges, label_value, edges_attr)
    
    return data

def preprocess_sentences(nlp, review):
    doc = nlp(review)
    sentences = [sent for sent in doc.sents]
    sentences = [{token.text.lower() : (token.pos_, token.dep_) for token in sent if not token.is_stop and token.is_alpha} for sent in sentences]
    return sentences

def get_biagrams(sentences):
    biagrams = []
    for sent in sentences:
        for i in range(len(sent)-1):
            biagrams.append((list(sent.keys())[i], list(sent.keys())[i+1]))
    return biagrams

def get_weighted_biagram_appearance(biagrams, sent, weights_each_type, method="directed_bi"):
    #methods : directed_bi, undirected_bi, weighted_bi
    dico_biagrams = {}
    gram_attribute = list(weights_each_type.keys())
    for biagram in biagrams:
        if method == "directed_bi":
            if biagram not in dico_biagrams:
                dico_biagrams[biagram] = np.zeros((len(gram_attribute), 2))
                idx_0 = gram_attribute.index(sent[biagram[0]][0])
                idx_1 = gram_attribute.index(sent[biagram[1]][0])
                dico_biagrams[biagram][idx_0][0] = 1
                dico_biagrams[biagram][idx_1][1] = 1 
                dico_biagrams[biagram] = dico_biagrams[biagram].flatten()
        elif method == "undirected_bi":
            if biagram not in dico_biagrams and (biagram[1], biagram[0]) not in dico_biagrams:
                dico_biagrams[biagram] = np.zeros(len(gram_attribute))
                idx_0 = gram_attribute.index(sent[biagram[0]][0])
                idx_1 = gram_attribute.index(sent[biagram[1]][0])
                dico_biagrams[biagram][idx_0] = 1
                dico_biagrams[biagram][idx_1] = 1 
        elif method == "weighted_bi":
            if biagram not in dico_biagrams and (biagram[1], biagram[0]) not in dico_biagrams:
                dico_biagrams[biagram] = weights_each_type[sent[biagram[0]][0]] + weights_each_type[sent[biagram[1]][0]]
            elif biagram in dico_biagrams:
                dico_biagrams[biagram] +=  weights_each_type[sent[biagram[0]][0]] + weights_each_type[sent[biagram[1]][0]]
            elif (biagram[1], biagram[0]) in dico_biagrams:
                dico_biagrams[(biagram[1], biagram[0])] += weights_each_type[sent[biagram[0]][0]] + weights_each_type[sent[biagram[1]][0]]
    return dico_biagrams

def biagrams_to_nx_graph(list_of_words, dico_biagrams):
    G = nx.Graph()
    ## nodes as words 
    G.add_nodes_from(list_of_words)
    ## add edges
    for biagram in dico_biagrams.keys():
        G.add_edge(biagram[0], biagram[1], weight = dico_biagrams[biagram])
    return G

def get_graph_features(G, nlp):    
    # Get the node features
    node_features = []
    for node in G.nodes():
        node_features.append(nlp.vocab[node].vector)
    node_features = np.array(node_features)
    # Get the edges
    edges = []
    for edge in G.edges():
        edges.append([list(G.nodes()).index(edge[0]), list(G.nodes()).index(edge[1])])
    edges = np.array(edges)
    ## edge_attr 
    edges_attr  = []
    for edge in G.edges():
        edges_attr.append([G.edges[edge]['weight']])
    edges_attr = np.array(edges_attr)
    return node_features, edges, edges_attr

def features_to_torch(node_features, edges, label_value, edges_attr):    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    y = torch.tensor(label_value, dtype=torch.float)
    edge_attr = torch.tensor(edges_attr, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr = edge_attr, y=y)
    return data