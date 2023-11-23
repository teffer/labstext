import spacy
import networkx as nx
import matplotlib.pyplot as plt
from spacy.lang.ru import Russian
from nltk.parse import DependencyGraph

def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def spacy_ner(text):
    nlp = spacy.load('ru_core_news_sm')
    doc = nlp(text)
    named_entities = []

    for ent in doc.ents:
        named_entities.append((ent.text, ent.label_))

    return named_entities

def convert_to_dependency_graph(text):
    nlp = spacy.load('ru_core_news_sm')
    doc = nlp(text)
    conll_format = []
    
    for sent in doc.sents:
        conll_format.append('\n'.join([f'{i+1}\t{token.text}\t{token.lemma_}\t{token.pos_}\t{token.tag_}\t_\t{token.head.i+1}\t{token.dep_}\t_\t_' for i, token in enumerate(sent)]))

    conll_text = '\n\n'.join(conll_format)
    return DependencyGraph(conll_text, top_relation_label='root')

def extract_triples(parsed_sentence):
    triples = []
    for node in parsed_sentence.nodes.values():
        if node['rel'] == 'root':
            subject = find_subject(node)
            relation = node['word']
            object_ = find_object(node)
            if subject and relation and object_:
                triples.append((subject, relation, object_))
    return triples

def find_subject(root_node):
    for child_index in root_node['deps']['nsubj']:
        return root_node['deps']['nsubj'][child_index]['word']
    return None

def find_object(root_node):
    for child_index in root_node['deps']['obj']:
        return root_node['deps']['obj'][child_index]['word']
    return None

def add_triple_to_graph(graph, triple):
    subject, relation, object_ = triple
    graph.add_node(subject)
    graph.add_node(object_)
    graph.add_edge(subject, object_, label=relation)

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    print("Nodes:", graph.nodes())
    print("Edges:", graph.edges())
    labels = nx.get_edge_attributes(graph, 'label')
    print(labels)
    nx.draw_networkx(graph, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.savefig('graph_lab8.png')


def start(file_path):
    text = load_text_from_file(file_path)
    
    named_entities = spacy_ner(text)
    print("Named Entities:", named_entities)

    spacy_dep_graph = convert_to_dependency_graph(text)

    triples = extract_triples(spacy_dep_graph)

    G = nx.Graph()
    for triple in triples:
        add_triple_to_graph(G, triple)
    visualize_graph(G)

if __name__ == '__main__':
    start('text.txt')