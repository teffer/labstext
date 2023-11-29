import spacy
import networkx as nx
from collections import Counter
nlp = spacy.load('en_core_web_lg')
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def sentence_segmentation(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def syntactic_analysis(sentence):
    doc = nlp(sentence)
    return doc

def named_entity_recognition(doc):
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def build_knowledge_graph(sentences):
    graph = nx.DiGraph()
    for sentence in sentences:
        doc = syntactic_analysis(sentence)
        obj=None
        subject =None
        for token in doc:
            if token.dep_ == 'ROOT':
                predicate = token.text
                obj=None
                subject =None
                for child in token.children:
                    if'nsubj' in child.dep_:
                        subject = child.text
            elif token.dep_ in ('dobj', 'obl','pobj'):
                obj = token.text
            if subject and obj:
                graph.add_node(subject)
                graph.add_node(obj)
                graph.add_edge(subject, obj, label=predicate)

    return graph

def visualize_graph(graph):
    nodes = [{'id': node, 'label': graph.nodes[node].get('label', node)} for node in graph.nodes]
    edges = [{'from': edge[0], 'to': edge[1], 'label': edge[2]['label']} for edge in graph.edges(data=True)]


    html_content = f"""
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    </head>
    <body>
        <div id="graph-container"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({nodes});
            var edges = new vis.DataSet({edges});

            var container = document.getElementById('graph-container');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            var options = {{}};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """

    with open('graph.html', 'w', encoding='utf-8') as file:
        file.write(html_content)

def search_in_graph(graph, query):
    if query in graph:
        return graph[query]
    else:
        return []

def start():
    file_path = 'text.txt'
    text = load_text(file_path)
    doc = nlp(text)
    entities = named_entity_recognition(doc)
    c = Counter(entities)
    res = [x for x in entities if c[x] == 1]
    counter = 0
    for entity in res:
        print(entity)
        text = text.replace(entity[0], f'entity_{counter}')
        counter += 1
    print(res)
    sentences = sentence_segmentation(text)
    knowledge_graph = build_knowledge_graph(sentences)
    visualize_graph(knowledge_graph)
    query = input("Поиск слова: ")
    results = search_in_graph(knowledge_graph, query)
    print(f"Объекты связанные с {query}: {results}")

if __name__ == '__main__':
    start()
