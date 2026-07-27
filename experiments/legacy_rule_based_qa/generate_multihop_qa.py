import csv
import json
import os
import random

import networkx as nx

DATA_DIR = "cora_dataset"
OUTPUT_FILE = os.path.join("sft_data", "05_multihop_qa.jsonl")
SEED = 42
N_SAMPLES = 500

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

def fmt_list(titles, limit=5):
    sample = titles[:limit]
    rest = len(titles) - limit
    s = ", ".join(f"<{t}>" for t in sample)
    return s + (f", and {rest} more" if rest > 0 else "")

def main():
    random.seed(SEED)
    os.makedirs("sft_data", exist_ok=True)
    papers, raw_edges, G = load_graph()
    nodes = list(papers)
    qa = []

    for nid in random.sample(nodes, N_SAMPLES):
        title = papers[nid]
        hop1 = list(G.successors(nid))
        if not hop1:
            continue
        hop2 = set()
        for h in hop1:
            hop2.update(G.successors(h))
        hop2 -= {nid}
        hop2 -= set(hop1)
        if not hop2:
            continue
        hop2_list = [papers[n] for n in hop2]
        qa.append({
            "query": (
                f"In the Cora dataset, what papers are cited by the papers "
                f"that <{title}> cites (2-hop out-neighborhood, excluding "
                f"direct citations)?"
            ),
            "answer": (
                f"<{title}> directly cites {len(hop1)} paper(s). "
                f"The papers those in turn cite (2-hop out-neighbors) include "
                f"{len(hop2_list)} paper(s): {fmt_list(hop2_list)}."
            ),
        })

    seen, count, tries = set(), 0, 0
    while count < N_SAMPLES and tries < N_SAMPLES * 20:
        a, b = random.choice(nodes), random.choice(nodes)
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            tries += 1
            continue
        seen.add(key)
        out_a = set(G.successors(a))
        out_b = set(G.successors(b))
        common = out_a & out_b
        ta, tb = papers[a], papers[b]
        if common:
            qa.append({
                "query": (
                    f"What papers are commonly cited by both <{ta}> and <{tb}> "
                    f"in the Cora dataset?"
                ),
                "answer": (
                    f"<{ta}> and <{tb}> both cite {len(common)} paper(s) in common: "
                    f"{fmt_list([papers[n] for n in common])}."
                ),
            })
        elif random.random() < 0.3:
            qa.append({
                "query": (
                    f"What papers are commonly cited by both <{ta}> and <{tb}> "
                    f"in the Cora dataset?"
                ),
                "answer": (
                    f"<{ta}> and <{tb}> have no commonly cited papers in the "
                    f"Cora dataset."
                ),
            })
        count += 1
        tries += 1

    seen, count, tries = set(), 0, 0
    while count < N_SAMPLES and tries < N_SAMPLES * 20:
        a, b = random.choice(nodes), random.choice(nodes)
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            tries += 1
            continue
        seen.add(key)
        in_a = set(G.predecessors(a))
        in_b = set(G.predecessors(b))
        common = in_a & in_b
        ta, tb = papers[a], papers[b]
        if common:
            qa.append({
                "query": (
                    f"What papers in the Cora dataset cite both <{ta}> and <{tb}>?"
                ),
                "answer": (
                    f"{len(common)} paper(s) cite both <{ta}> and <{tb}>: "
                    f"{fmt_list([papers[n] for n in common])}."
                ),
            })
        elif random.random() < 0.3:
            qa.append({
                "query": (
                    f"What papers in the Cora dataset cite both <{ta}> and <{tb}>?"
                ),
                "answer": (
                    f"No paper in the Cora dataset cites both <{ta}> and <{tb}>."
                ),
            })
        count += 1
        tries += 1

    count, tries = 0, 0
    while count < N_SAMPLES and tries < N_SAMPLES * 20:
        a, b = random.choice(nodes), random.choice(nodes)
        if a == b:
            tries += 1
            continue
        intermediates = set(G.successors(a)) & set(G.predecessors(b))
        ta, tb = papers[a], papers[b]
        if intermediates:
            mid = random.choice(list(intermediates))
            tm = papers[mid]
            qa.append({
                "query": (
                    f"Is there a 2-hop directed citation path from <{ta}> to <{tb}> "
                    f"in the Cora dataset? If so, name an intermediate paper."
                ),
                "answer": (
                    f"Yes. One 2-hop path is: <{ta}> → <{tm}> → <{tb}>. "
                    f"(There are {len(intermediates)} intermediate paper(s) in total.)"
                ),
            })
            count += 1
        elif random.random() < 0.2:
            qa.append({
                "query": (
                    f"Is there a 2-hop directed citation path from <{ta}> to <{tb}> "
                    f"in the Cora dataset?"
                ),
                "answer": (
                    f"No. There is no paper that <{ta}> cites and that also cites "
                    f"<{tb}>, so no 2-hop directed path exists between them."
                ),
            })
            count += 1
        tries += 1

    count, tries = 0, 0
    while count < 300 and tries < 3000:
        a, b = random.choice(nodes), random.choice(nodes)
        if a == b:
            tries += 1
            continue
        in_a = set(G.predecessors(a))
        out_b = set(G.successors(b))
        overlap = in_a & out_b
        ta, tb = papers[a], papers[b]
        if overlap:
            qa.append({
                "query": (
                    f"In the Cora dataset, which papers both cite <{ta}> "
                    f"and are cited by <{tb}>?"
                ),
                "answer": (
                    f"{len(overlap)} paper(s) both cite <{ta}> and are cited by "
                    f"<{tb}>: {fmt_list([papers[n] for n in overlap])}."
                ),
            })
            count += 1
        tries += 1

    random.shuffle(qa)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[multihop]     {len(qa):>6,} pairs  →  {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
