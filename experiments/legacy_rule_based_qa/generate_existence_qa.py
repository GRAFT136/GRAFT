import csv
import json
import os
import random

import networkx as nx

DATA_DIR = "cora_dataset"
OUTPUT_FILE = os.path.join("sft_data", "01_existence_qa.jsonl")
SEED = 42
N_POS = 3000
N_NEG = 3000
N_CONN = 500

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
        f"Edge direction likely wrong: max in-degree={max_in} is too low. "
        "Expected >50 for classic papers like Goldberg/Sutton."
    )
    return papers, edges, G

def main():
    random.seed(SEED)
    os.makedirs("sft_data", exist_ok=True)
    papers, raw_edges, G = load_graph()
    edge_set = set(raw_edges)
    nodes = list(papers)
    qa = []

    q_pos = [
        "Does paper <{s}> directly cite paper <{t}>?",
        "In the Cora dataset, is there a direct citation from <{s}> to <{t}>?",
        "Does <{s}> reference <{t}> in its bibliography?",
        "Is <{t}> listed as a reference of <{s}>?",
    ]
    for src, tgt in random.sample(raw_edges, min(N_POS, len(raw_edges))):
        s, t = papers[src], papers[tgt]
        qa.append({
            "query": random.choice(q_pos).format(s=s, t=t),
            "answer": f"Yes. <{s}> directly cites <{t}> in the Cora citation network.",
        })

    q_neg = [
        "Does paper <{s}> directly cite paper <{t}>?",
        "In the Cora dataset, is there a direct citation from <{s}> to <{t}>?",
        "Does <{s}> reference <{t}> in its bibliography?",
        "Is <{t}> listed as a reference of <{s}>?",
    ]
    neg, tries = 0, 0
    while neg < N_NEG and tries < N_NEG * 30:
        sid, tid = random.choice(nodes), random.choice(nodes)
        if sid != tid and (sid, tid) not in edge_set:
            s, t = papers[sid], papers[tid]
            qa.append({
                "query": random.choice(q_neg).format(s=s, t=t),
                "answer": f"No. <{s}> does not directly cite <{t}> in the Cora citation network.",
            })
            neg += 1
        tries += 1

    for nid in nodes:
        title = papers[nid]
        out_deg = G.out_degree(nid)
        succs = [papers[n] for n in G.successors(nid)]
        if out_deg > 0:
            ex = ", ".join(f"<{p}>" for p in succs[:3])
            qa.append({
                "query": f"Does paper <{title}> cite any other papers in the Cora dataset?",
                "answer": f"Yes. <{title}> cites {out_deg} paper(s). Examples include: {ex}.",
            })
        else:
            qa.append({
                "query": f"Does paper <{title}> cite any other papers in the Cora dataset?",
                "answer": f"No. <{title}> does not cite any other papers in the Cora citation network.",
            })

    for nid in nodes:
        title = papers[nid]
        in_deg = G.in_degree(nid)
        preds = [papers[n] for n in G.predecessors(nid)]
        if in_deg > 0:
            ex = ", ".join(f"<{p}>" for p in preds[:3])
            qa.append({
                "query": f"Has paper <{title}> been cited by any other paper in the Cora dataset?",
                "answer": f"Yes. <{title}> has been cited by {in_deg} paper(s). Papers that cite it include: {ex}.",
            })
        else:
            qa.append({
                "query": f"Has paper <{title}> been cited by any other paper in the Cora dataset?",
                "answer": f"No. <{title}> has not been cited by any other paper in the Cora citation network.",
            })

    G_und = G.to_undirected()
    for _ in range(N_CONN):
        a, b = random.choice(nodes), random.choice(nodes)
        if a == b:
            continue
        ta, tb = papers[a], papers[b]
        connected = nx.has_path(G_und, a, b)
        if connected:
            qa.append({
                "query": f"Are papers <{ta}> and <{tb}> connected in the Cora citation network?",
                "answer": f"Yes. <{ta}> and <{tb}> are connected in the Cora citation network (directly or via intermediate papers).",
            })
        else:
            qa.append({
                "query": f"Are papers <{ta}> and <{tb}> connected in the Cora citation network?",
                "answer": f"No. <{ta}> and <{tb}> are not connected in the Cora citation network.",
            })

    is_dag = nx.is_directed_acyclic_graph(G)
    qa.extend([
        {
            "query": "Does the Cora citation network contain any directed cycles?",
            "answer": (
                "Yes. The Cora citation network contains directed cycles, meaning some papers mutually cite each other in a circular fashion."
                if not is_dag else
                "No. The Cora citation network is a directed acyclic graph (DAG): there are no circular citation chains."
            ),
        },
        {
            "query": "Is the Cora citation graph a directed acyclic graph (DAG)?",
            "answer": (
                "No. The Cora citation graph is not a DAG because it contains at least one directed cycle."
                if not is_dag else
                "Yes. The Cora citation graph is a directed acyclic graph (DAG), meaning there are no circular citation dependencies."
            ),
        },
    ])

    random.shuffle(qa)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[existence]    {len(qa):>6,} pairs  →  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
