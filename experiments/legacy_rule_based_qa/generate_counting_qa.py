import csv
import json
import os
import random

import networkx as nx

DATA_DIR = "cora_dataset"
OUTPUT_FILE = os.path.join("sft_data", "02_counting_qa.jsonl")
SEED = 42

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
    nodes = list(papers)
    qa = []

    q_out = [
        "How many papers does <{t}> cite in the Cora dataset?",
        "What is the number of papers referenced by <{t}>?",
        "How many outgoing citations does <{t}> have in the Cora citation network?",
        "What is the out-degree of paper <{t}> in the Cora graph?",
    ]
    for nid in nodes:
        title = papers[nid]
        out_deg = G.out_degree(nid)
        qa.append({
            "query": random.choice(q_out).format(t=title),
            "answer": f"<{title}> cites {out_deg} paper(s) in the Cora dataset (out-degree = {out_deg}).",
        })

    q_in = [
        "How many papers cite <{t}> in the Cora dataset?",
        "What is the number of papers that reference <{t}>?",
        "How many incoming citations does <{t}> have in the Cora citation network?",
        "What is the in-degree of paper <{t}> in the Cora graph?",
    ]
    for nid in nodes:
        title = papers[nid]
        in_deg = G.in_degree(nid)
        qa.append({
            "query": random.choice(q_in).format(t=title),
            "answer": f"<{title}> is cited by {in_deg} paper(s) in the Cora dataset (in-degree = {in_deg}).",
        })

    q_deg = [
        "What is the total degree of paper <{t}> in the Cora citation network?",
        "How many direct citation relationships does <{t}> have (counting both directions)?",
    ]
    for nid in nodes:
        title = papers[nid]
        in_deg = G.in_degree(nid)
        out_deg = G.out_degree(nid)
        total = in_deg + out_deg
        qa.append({
            "query": random.choice(q_deg).format(t=title),
            "answer": (
                f"<{title}> has a total degree of {total} "
                f"(in-degree: {in_deg}, out-degree: {out_deg})."
            ),
        })

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    qa.extend([
        {
            "query": "How many papers are in the Cora dataset?",
            "answer": f"The Cora dataset contains {n_nodes} papers (nodes).",
        },
        {
            "query": "What is the total number of papers in the Cora citation network?",
            "answer": f"There are {n_nodes} papers in total in the Cora citation network.",
        },
        {
            "query": "How many citation relationships (edges) exist in the Cora dataset?",
            "answer": f"The Cora dataset contains {n_edges} directed citation relationships (edges).",
        },
        {
            "query": "What is the total number of edges in the Cora citation graph?",
            "answer": f"There are {n_edges} directed edges in the Cora citation graph.",
        },
    ])

    avg = n_edges / n_nodes
    qa.extend([
        {
            "query": "What is the average number of papers cited per paper in the Cora dataset?",
            "answer": (
                f"The average out-degree in the Cora citation network is {avg:.2f}, "
                f"meaning each paper cites approximately {avg:.2f} other papers on average."
            ),
        },
        {
            "query": "What is the average in-degree of papers in the Cora dataset?",
            "answer": (
                f"The average in-degree is {avg:.2f}, "
                f"meaning each paper is cited by approximately {avg:.2f} other papers on average."
            ),
        },
        {
            "query": "What is the average degree of nodes in the Cora citation graph (undirected)?",
            "answer": (
                f"In the undirected Cora citation graph, the average degree is "
                f"{2 * n_edges / n_nodes:.2f} (each undirected edge contributes 2 to the degree sum)."
            ),
        },
    ])

    max_in_id  = max(nodes, key=lambda n: G.in_degree(n))
    max_out_id = max(nodes, key=lambda n: G.out_degree(n))
    min_in_id  = min(nodes, key=lambda n: G.in_degree(n))
    min_out_id = min(nodes, key=lambda n: G.out_degree(n))

    qa.extend([
        {
            "query": "Which paper in the Cora dataset has the most incoming citations?",
            "answer": (
                f"The paper with the highest in-degree is <{papers[max_in_id]}>, "
                f"which is cited by {G.in_degree(max_in_id)} other papers."
            ),
        },
        {
            "query": "Which paper in the Cora dataset cites the most other papers?",
            "answer": (
                f"The paper with the highest out-degree is <{papers[max_out_id]}>, "
                f"which cites {G.out_degree(max_out_id)} other papers."
            ),
        },
        {
            "query": "Which paper in the Cora dataset has the fewest incoming citations?",
            "answer": (
                f"The minimum in-degree is {G.in_degree(min_in_id)}. "
                f"One such paper is <{papers[min_in_id]}>."
            ),
        },
        {
            "query": "Which paper in the Cora dataset cites the fewest other papers?",
            "answer": (
                f"The minimum out-degree is {G.out_degree(min_out_id)}. "
                f"One such paper is <{papers[min_out_id]}>."
            ),
        },
    ])

    random.shuffle(qa)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[counting]     {len(qa):>6,} pairs  →  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
