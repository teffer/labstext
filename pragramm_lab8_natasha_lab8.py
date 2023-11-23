from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
import networkx as nx
import json
import matplotlib.pyplot as plt

def natasha_ner(text):
    embedding = NewsEmbedding()
    ner_tagger = NewsNERTagger(embedding)
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    named_entities = [(ent.text, ent.type) for ent in doc.ner]
    return named_entities

def extract_subject_object_relation(tokens):
    triples = []

    for token in tokens:
        if "subj" in token.rel or "obj" in token.rel:
            subject = token.text
            relation = token.head.text
            object_ = find_object(token.head)
            if object_:
                triples.append((subject, relation, object_))

    return triples

def find_object(head_token):
    for child in head_token.children:
        if "obj" in child.rel:
            return child.text
    return None

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

def visualize_graph(visjs_data, output_file):
    G = nx.Graph()

    for edge in visjs_data["edges"]:
        G.add_node(edge["from"])
        G.add_node(edge["to"])
        G.add_edge(edge["from"], edge["to"], label=edge["label"])

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'label')

    nx.draw_networkx(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.savefig(output_file)

def process_text_and_build_graphs(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    for i, sent in enumerate(doc.sents):
        print(f"Processing Sentence {i + 1}: {sent.text}")
        triples = extract_subject_object_relation(sent.tokens)
        visjs_data = create_visjs_data(triples)
        visualize_graph(visjs_data, f'graph_sentence_{i + 1}.png')

if __name__ == '__main__':
    segmenter = Segmenter()
    embedding = NewsEmbedding()
    ner_tagger = NewsNERTagger(embedding)

    text = """
    Дерсу никогда не стрелял в живое понапрасну.
    Если продовольствия в отряде хватало, он, имея возможность добыть нескольких изюбрей, мог ограничиться рябчиком.
    Когда один из солдат забавы ради прицелился в ворону, Дерсу его остановил.
    Арсеньев говорил, что к идеям охраны природы, разумного пользования её дарами дикарь Дерсу стоял куда ближе многих европейцев, слывущих людьми образованными и культурными.
    """

    process_text_and_build_graphs(text)
