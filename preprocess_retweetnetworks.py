### ================== RETWEET NETWORK PREPROCESSING ================== ###
###
### Takes each retweet network file with columns ['source','target','weight']
### and creates two files in the same folder:
### - {filename}_rtn_pp.csv: preprocessed retweet network
### - {filename}_rtn_stats.csv: stats file
###
### Preprocessing:
### - Reduce to giant component
### - Remove nodes with only 1 out-neighbor and in-degree 0
### - Discard graphs with n_nodes < 50 after these two steps
### 
### Stats:
### n_nodes_full, n_nodes_gc, n_nodes_sa

import os
import csv
import pandas as pd
import igraph as ig
from tqdm import tqdm

## variables
RTN_DIR = "./data/rtn/"
OUT_DIR = "./output/rtn/"

def create_dir_if_not_exist(path):
    if not os.path.exists(path): 
        os.makedirs(path)

create_dir_if_not_exist(OUT_DIR)

## constants
RTN_FIELDS = ["source","target","weight"]
STATS_FIELDS = ["N_nodes_full","N_nodes_gc","N_nodes_sa"]
tweet_dtypes = {'source':str,
                'target':str,
                'weight':int,}

## functions
def load_rtn(path):
    df = pd.read_csv(path,dtype=tweet_dtypes)
    return df

def extract_giant_component_igraph(igraph_graph):
    return igraph_graph.components(mode="weak").giant()
def soft_aggregation_igraph(igraph_graph):
    g = igraph_graph.copy()
    todel = []
    for v in g.vs:
        if g.degree(v, mode="in") == 0 and len(set(g.neighbors(v, mode="out"))) < 2:
            todel.append(v.index)
    g.delete_vertices(todel)
    return g

rtn_files = os.listdir(RTN_DIR)
rtn_files = sorted([RTN_DIR+i for i in rtn_files if i[-8:] == "_rtn.csv"])

# print(len(rtn_files))

print("Preprocessing retweet networks...")
for idx in tqdm(range(len(rtn_files))):
    file = rtn_files[idx]
    outfile = file.replace("_rtn.csv","_rtn_pp.csv")
    statsfile = file.replace("_rtn.csv","_stats.csv")
    
    outfile = outfile.replace(RTN_DIR,OUT_DIR)
    statsfile = statsfile.replace(RTN_DIR,OUT_DIR)

    ## load csv
    df = load_rtn(file)

    ## build igraph object
    G = ig.Graph.TupleList(df.itertuples(index=False), 
                                   directed=True, 
                                   weights=True,
                                   )
    
    N_nodes_sa = 0
    N_nodes_gc = 0
    
    if len (G.vs) >= 2:

        ## remove self-loops
        G = G.simplify(multiple=False,loops=True)
        N_nodes_full = len(G.vs)
        
        ## giant component
        G = extract_giant_component_igraph(G)
        N_nodes_gc = len(G.vs)
        gc_size = N_nodes_gc / N_nodes_full

        ## soft aggregation
        G = soft_aggregation_igraph(G)
        G = extract_giant_component_igraph(G)
        N_nodes_sa = len(G.vs)
        
        # print("GC size:",gc_size)
        # print("N_nodes_init:",N_nodes_full)
        # print("N_nodes_gc",N_nodes_gc)
        # print("N_nodes_sa:",N_nodes_sa)
        
        if N_nodes_sa >= 50:
            
            ## write the rtn_pp file
            with open (outfile,"w",encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(RTN_FIELDS)
                for e in G.es:
                    user_id = G.vs['name'][e.source]
                    retweeted_user_id = G.vs['name'][e.target]
                    csv_row = [user_id,retweeted_user_id,e['weight']]
                    w.writerow(csv_row)
        else:
            pass
        
    else:
        pass

    ## write the stats file
    with open(statsfile,"w",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(STATS_FIELDS)
        statsrow = [N_nodes_full,N_nodes_gc,N_nodes_sa]
        w.writerow(statsrow)

