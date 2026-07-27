import csv
import json
import os
import random

import networkx as nx

DATA_DIR = "cora_dataset"
OUTPUT_FILE = os.path.join("sft_data", "04_substructure_qa.jsonl")
SEED = 42
N_TRI_POS  = 1000
N_TRI_NEG  = 1000
N_MUT_NEG  = 500

def load_graph():
    papers = {}
    with open(os.path.join(DATA_DIR, "all.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            papers[int(row["id"])] = row["T"].strip()
    edges = []
    with open(os.path.join(DATA_DIR, "edges.csv"), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tcol = next(k for k in reader.fieldnames if k.startswith("target"))
        for row in reader:
            edges.append((int(row[tcol]), int(row["source"])))
    G = nx.DiGraph()
    G.add_nodes_from(papers)
    G.add_edges_from(edges)
    max_in = max(dict(G.in_degree()).values())
    assert max_in > 50, (
        f"Edge direction likely wrong: max in-degree={max_in}. Expected >50."
    )
    return papers, edges, G

def main():
    random.seed(SEED)
    os.makedirs("sft_data", exist_ok=True)
    papers, raw_edges, G = load_graph()
    edge_set = set(raw_edges)
    nodes = list(papers)
    G_und = G.to_undirected()
    qa = []

    tri_counts = nx.triangles(G_und)
    for nid in nodes:
        title = papers[nid]
        count = tri_counts[nid]
        if count > 0:
            qa.append({
                "query": (
                    f"How many triangles does paper <{title}> participate in "
                    f"(in the undirected Cora citation graph)?"
                ),
                "answer": (
                    f"<{title}> participates in {count} triangle(s) in the "
                    f"undirected Cora citation network."
                ),
            })
        else:
            if random.random() < 0.2:
                qa.append({
                    "query": (
                        f"How many triangles does paper <{title}> participate in "
                        f"(in the undirected Cora citation graph)?"
                    ),
                    "answer": (
                        f"<{title}> does not participate in any triangles in the "
                        f"undirected Cora citation network."
                    ),
                })

    tri_pos, tries = [], 0
    while len(tri_pos) < N_TRI_POS and tries < N_TRI_POS * 30:
        a = random.choice(nodes)
        nbrs = list(G_und.neighbors(a))
        if len(nbrs) >= 2:
            b, c = random.sample(nbrs, 2)
            if G_und.has_edge(b, c):
                tri_pos.append((a, b, c))
        tries += 1

    for a, b, c in tri_pos:
        ta, tb, tc = papers[a], papers[b], papers[c]
        qa.append({
            "query": (
                f"Do papers <{ta}>, <{tb}>, and <{tc}> form a triangle "
                f"in the Cora citation network?"
            ),
            "answer": (
                f"Yes. <{ta}>, <{tb}>, and <{tc}> form a triangle in the "
                f"undirected Cora citation network: there is a citation relationship "
                f"between each pair of these three papers."
            ),
        })

    tri_neg, tries = [], 0
    while len(tri_neg) < N_TRI_NEG and tries < N_TRI_NEG * 30:
        a = random.choice(nodes)
        nbrs = list(G_und.neighbors(a))
        if len(nbrs) >= 2:
            b, c = random.sample(nbrs, 2)
            if not G_und.has_edge(b, c):
                tri_neg.append((a, b, c))
        tries += 1

    for a, b, c in tri_neg:
        ta, tb, tc = papers[a], papers[b], papers[c]
        qa.append({
            "query": (
                f"Do papers <{ta}>, <{tb}>, and <{tc}> form a triangle "
                f"in the Cora citation network?"
            ),
            "answer": (
                f"No. <{ta}>, <{tb}>, and <{tc}> do not form a triangle. "
                f"While <{ta}> has a citation relationship with both <{tb}> and <{tc}>, "
                f"there is no direct citation relationship between <{tb}> and <{tc}>."
            ),
        })

    mutual_pairs = [(u, v) for u, v in raw_edges if (v, u) in edge_set and u < v]
    for u, v in mutual_pairs:
        tu, tv = papers[u], papers[v]
        qa.append({
            "query": f"Is there a mutual citation relationship between <{tu}> and <{tv}>?",
            "answer": (
                f"Yes. <{tu}> and <{tv}> mutually cite each other: "
                f"<{tu}> cites <{tv}> and <{tv}> cites <{tu}>."
            ),
        })

    mut_neg, tries = [], 0
    while len(mut_neg) < min(N_MUT_NEG, len(mutual_pairs)) and tries < N_MUT_NEG * 20:
        sid, tid = random.choice(nodes), random.choice(nodes)
        if (sid, tid) in edge_set and (tid, sid) not in edge_set:
            mut_neg.append((sid, tid))
        tries += 1

    for u, v in mut_neg:
        tu, tv = papers[u], papers[v]
        qa.append({
            "query": f"Is there a mutual citation relationship between <{tu}> and <{tv}>?",
            "answer": (
                f"No. While <{tu}> cites <{tv}>, <{tv}> does not cite <{tu}> back. "
                f"The citation is one-directional."
            ),
        })

    avg_in = G.number_of_edges() / G.number_of_nodes()
    hub_thresh = avg_in * 5
    for nid in nodes:
        title = papers[nid]
        in_deg = G.in_degree(nid)
        if in_deg >= hub_thresh:
            qa.append({
                "query": f"Is paper <{title}> a citation hub in the Cora dataset?",
                "answer": (
                    f"Yes. <{title}> is a citation hub with {in_deg} incoming citations, "
                    f"significantly above the network average of {avg_in:.1f}."
                ),
            })
        elif in_deg == 0 and random.random() < 0.15:
            qa.append({
                "query": f"Is paper <{title}> a citation hub in the Cora dataset?",
                "answer": (
                    f"No. <{title}> has no incoming citations (in-degree = 0) "
                    f"and is not a hub."
                ),
            })

    for nid in random.sample(nodes, 300):
        title = papers[nid]
        exclusive = [
            p for p in G.predecessors(nid)
            if G.out_degree(p) == 1
        ]
        if len(exclusive) >= 3:
            ex = ", ".join(f"<{papers[p]}>" for p in exclusive[:3])
            qa.append({
                "query": (
                    f"Does paper <{title}> act as the center of a star pattern "
                    f"in the Cora citation network (cited by papers that cite only it)?"
                ),
                "answer": (
                    f"Yes. <{title}> forms a star pattern: {ex} "
                    f"(and {len(exclusive) - 3} more) each cite only <{title}> "
                    f"and no other papers."
                ),
            })

    random.shuffle(qa)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[substructure] {len(qa):>6,} pairs  →  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
