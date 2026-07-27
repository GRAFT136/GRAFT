import csv
import json
import os
import random

import networkx as nx

DATA_DIR = "cora_dataset"
OUTPUT_FILE = os.path.join("sft_data", "03_traversal_qa.jsonl")
SEED = 42
N_REACH = 500
N_PATH  = 500
N_TRAV  = 200
N_KHOP  = 200

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

def path_str(path, papers):
    return " → ".join(f"<{papers[n]}>" for n in path)

def main():
    random.seed(SEED)
    os.makedirs("sft_data", exist_ok=True)
    papers, raw_edges, G = load_graph()
    nodes = list(papers)
    qa = []

    for _ in range(N_REACH):
        src, tgt = random.choice(nodes), random.choice(nodes)
        if src == tgt:
            continue
        st, tt = papers[src], papers[tgt]
        try:
            path = nx.shortest_path(G, src, tgt)
            dist = len(path) - 1
            qa.append({
                "query": (
                    f"Is paper <{tt}> reachable from paper <{st}> "
                    f"through directed citation links in the Cora dataset?"
                ),
                "answer": (
                    f"Yes. <{tt}> is reachable from <{st}> in {dist} hop(s) "
                    f"via the directed citation path: {path_str(path, papers)}."
                ),
            })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            qa.append({
                "query": (
                    f"Is paper <{tt}> reachable from paper <{st}> "
                    f"through directed citation links in the Cora dataset?"
                ),
                "answer": (
                    f"No. There is no directed citation path from <{st}> to <{tt}> "
                    f"in the Cora dataset."
                ),
            })

    connected_pairs = []
    for _ in range(N_PATH * 20):
        if len(connected_pairs) >= N_PATH:
            break
        src, tgt = random.choice(nodes), random.choice(nodes)
        if src != tgt and nx.has_path(G, src, tgt):
            connected_pairs.append((src, tgt))

    q_path = [
        "What is the shortest directed citation path from <{s}> to <{t}> in the Cora dataset?",
        "How does the citation chain lead from <{s}> to <{t}> in Cora?",
        "What is the shortest sequence of papers connecting <{s}> to <{t}> via citations?",
    ]
    for src, tgt in connected_pairs:
        path = nx.shortest_path(G, src, tgt)
        dist = len(path) - 1
        st, tt = papers[src], papers[tgt]
        qa.append({
            "query": random.choice(q_path).format(s=st, t=tt),
            "answer": (
                f"The citation distance from <{st}> to <{tt}> is {dist} hop(s). "
                f"One shortest path: {path_str(path, papers)}."
            ),
        })

    for src, tgt in random.sample(connected_pairs, min(300, len(connected_pairs))):
        dist = nx.shortest_path_length(G, src, tgt)
        st, tt = papers[src], papers[tgt]
        qa.append({
            "query": (
                f"What is the citation distance (shortest directed path length) "
                f"between <{st}> and <{tt}> in the Cora dataset?"
            ),
            "answer": (
                f"The shortest directed citation distance from <{st}> to <{tt}> "
                f"is {dist} hop(s)."
            ),
        })

    for nid in random.sample(nodes, N_KHOP):
        title = papers[nid]
        hop1 = list(G.successors(nid))
        if not hop1:
            continue
        ex = ", ".join(f"<{papers[n]}>" for n in hop1[:5])
        suffix = f", and {len(hop1) - 5} more" if len(hop1) > 5 else ""
        qa.append({
            "query": (
                f"What papers are directly reachable (1-hop out-neighborhood) "
                f"from <{title}> in the Cora citation network?"
            ),
            "answer": (
                f"<{title}> directly cites {len(hop1)} paper(s): "
                f"{ex}{suffix}."
            ),
        })

        hop2 = set()
        for h in hop1:
            hop2.update(G.successors(h))
        hop2 -= {nid}
        hop2 -= set(hop1)
        if hop2:
            ex2 = ", ".join(f"<{papers[n]}>" for n in list(hop2)[:5])
            rest2 = max(0, len(hop2) - 5)
            suffix2 = f", and {rest2} more" if rest2 > 0 else ""
            qa.append({
                "query": (
                    f"What papers are 2 hops away (via outgoing citations) "
                    f"from <{title}> in the Cora citation network?"
                ),
                "answer": (
                    f"The 2-hop out-neighborhood of <{title}> (papers cited by its "
                    f"direct citations, excluding direct citations themselves) contains "
                    f"{len(hop2)} paper(s): {ex2}{suffix2}."
                ),
            })

    random.shuffle(qa)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[traversal]    {len(qa):>6,} pairs  →  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
