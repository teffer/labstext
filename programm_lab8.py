import spacy
import networkx as nx
import matplotlib.pyplot as plt
from spacy.lang.ru import Russian
import json
from natasha import Doc, NewsEmbedding, NewsNERTagger
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def spacy_ner(text):
    nlp = spacy.load('ru_core_news_lg')
    doc = nlp(text)
    named_entities = []

    for ent in doc.ents:
        named_entities.append((ent.text, ent.label_))

    return named_entities

def create_visjs_data(triples):
    nodes = []
    edges = []

    for triple in triples:
        subject, relation, object_ = triple
        nodes.extend([subject, object_])
        edges.append({"from": subject, "to": object_, "label": relation})

    unique_nodes = list(set(nodes))

    nodes_data = [{"id": node, "label": node} for node in unique_nodes]

    return {"nodes": nodes_data, "edges": edges}

def visualize_graph_js(visjs_data):
    template = """
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    </head>
    <body>
      <div id="graph"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet(%s);
        var edges = new vis.DataSet(%s);
        var container = document.getElementById('graph');
        var data = {
          nodes: nodes,
          edges: edges
        };
        var options = {};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """

    with open('graph.html', 'w', encoding='utf-8') as f:
        f.write(template % (json.dumps(visjs_data["nodes"]), json.dumps(visjs_data["edges"])))
def visualize_graph_js_multi_santence(visjs_data, output_file):
    template = """
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    </head>
    <body>
      <div id="graph"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet(%s);
        var edges = new vis.DataSet(%s);
        var container = document.getElementById('graph');
        var data = {
          nodes: nodes,
          edges: edges
        };
        var options = {};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(template % (json.dumps(visjs_data["nodes"]), json.dumps(visjs_data["edges"])))
def search_in_graph(graph, target_node):
    if target_node in graph:
        neighbors = list(graph.neighbors(target_node))
        if neighbors:
            print(f"К узлу '{target_node}':")
            for neighbor in neighbors:
                relation = graph[target_node][neighbor]['label']
                print(f"{neighbor} (Вид связи: {relation})")
        else:
            print(f"К узлу'{target_node} нет связей'.")
    else:
        print(f"Узла '{target_node}' не существует.")

def extract_subject_object_relation(doc):
    triples = []

    for token in doc:
        if "subj" in token.dep_:
            subject = token.text
            relation = token.head.text
            object_ = find_object(token.head)
            print(token.text, token.dep_, token.head.text, token.head.dep_)
            if object_:
                triples.append((subject, relation, object_))

    return triples

def find_object(head_token):
    for child in head_token.children:
        if "obj" in child.dep_:
            return child.text
        nested_object = find_object(child)
        if nested_object:
            return nested_object
    return None

def visualize_graph(triples):
    G = nx.Graph()

    for triple in triples:
        subject, relation, object_ = triple
        G.add_node(subject)
        G.add_node(object_)
        G.add_edge(subject, object_, label=relation)

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')

    nx.draw_networkx(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.savefig('graph_lab8.png')

def start(file_path):
    text = load_text_from_file(file_path)
    
    named_entities = spacy_ner(text)
    print("Named Entities:", named_entities)

    doc = nlp(text)
    triples = extract_subject_object_relation(doc)
    print(triples)
    visualize_graph(triples)
    visjs_data = create_visjs_data(triples)
    visualize_graph_js(visjs_data)
    for i, sent in enumerate(doc.sents):
        print(f"Обработка предложения {i + 1}: {sent.text}")
        triples = extract_subject_object_relation(sent)
        visjs_data = create_visjs_data(triples)
        print(visjs_data)
        visualize_graph_js_multi_santence(visjs_data, f'Граф_предложения_{i + 1}.html')
if __name__ == '__main__':
    nlp = spacy.load('ru_core_news_lg')
    start('text.txt')