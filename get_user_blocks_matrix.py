### ================== USER SUMMARY ================== ###
###
### Takes each sbm clustering from 'output/sbm' and merges the assignments into
### one big CSV that summarizes the clustering with the following columns:
###
### - user_id 
### - trends: '|'-delimited list of trends in which the user participated
### - in-degrees: '|'-delimited list of in-degrees of user in the same order as 'trends'
### - out-degrees: '|'-delimited list of out-degrees of user in the same order as 'trends'
### - blocks: '|'-delimited list of block assignment of user in the same order as 'trends'
### - in_degree_total
### - out_degree_total
### - N_trends_total

import os
import pandas as pd
import igraph as ig
from tqdm import tqdm

SBM_DIR = "./output/sbm/"
RTN_DIR = "./data/rtn/"
OUTFILE = "./output/users_blocks.csv"

# sbm_files = os.listdir(SBM_DIR)
# sbm_files = sorted([SBM_DIR+i for i in sbm_files if "_sbm.csv" in i])

sbm_stats = pd.read_csv("./output/sbm/sbm_stats.csv")
tns = sbm_stats['trend']
sbm_files = []
for t in tns:
    corfile = SBM_DIR+t+"_sbm_corr.csv"
    if os.path.isfile(corfile):
        sbm_files.append(SBM_DIR+t+"_sbm_corr.csv")
    else:
        sbm_files.append(SBM_DIR+t+"_sbm.csv")
sbm_files = sorted(sbm_files)        

# RTN_FIELDS = ["id","retweeted_id","user_id","retweeted_user_id","timestamp_utc","retweeted_timestamp_utc"]
RTN_FIELDS = ["source","target","weight"]
STATS_FIELDS = ["N_nodes_full","N_nodes_gc","N_nodes_sa"]
tweet_dtypes = {'weight':int,
                'source':str,
                'target':str,
                'retweeted_id':str,
                'user_id':str,
                'retweeted_user_id':str,
                'timestamp_utc':int,
                'retweeted_timestamp_utc':int,
                'block':int}

def extract_giant_component_igraph(igraph_graph):
    return igraph_graph.components(mode="weak").giant()

print("Extracting user blocks...")
## define a dict that we will transform to the final csv 
userdict = {}

for trend_idx in tqdm(range(len(sbm_files))):
    
    ## load the sbm
    sbm_file = sbm_files[trend_idx]
    sbm = pd.read_csv(sbm_file,dtype=tweet_dtypes)
    sbm = sbm.set_index('user_id')

    ## set the name for writing everything
    # trendname = sbm_file.replace(SBM_DIR,"").replace("_sbm.csv","")    
    trendname = sbm_file.replace(SBM_DIR,"").replace("_sbm","").replace(".csv","").replace("_corr","")    
    
    ## load the original retweet network
    rtn_file = RTN_DIR+trendname+"_rtn.csv"
    rtn = pd.read_csv(rtn_file,dtype=tweet_dtypes)
#     rtn = rtn[['user_id','retweeted_user_id']].rename(columns={'user_id':'source','retweeted_user_id':'target'})
    G = ig.Graph.TupleList(rtn.itertuples(index=False), directed=True, weights=True)
    G = G.simplify(multiple=False,loops=True)
    G = extract_giant_component_igraph(G)
    in_degrees = G.degree(mode='in')
    out_degrees = G.degree(mode='out')

    for u_idx,v in enumerate(G.vs):
        name = v['name']
        in_degree = in_degrees[u_idx]
        out_degree = out_degrees[u_idx]
        try:
            block = sbm.loc[name]['block']
        except KeyError:
            parent = G.neighbors(v,mode='out')[0]
            parent_name = G.vs['name'][parent]
            block = sbm.loc[parent_name]['block']

        try:
            userdict[name]['trends'] += f"|{trendname}"
            userdict[name]['in_degrees'] += f"|{in_degree}"
            userdict[name]['out_degrees'] += f"|{out_degree}"
            userdict[name]['blocks'] += f"|{int(block)}"
            userdict[name]['in_degree_total'] += in_degree
            userdict[name]['out_degree_total'] += out_degree
            userdict[name]['N_trends_total'] += 1
        except KeyError:
            userdict[name] = {}
            userdict[name]['trends'] = f"{trendname}"
            userdict[name]['in_degrees'] = f"{in_degree}"
            userdict[name]['out_degrees'] = f"{out_degree}"
            userdict[name]['blocks'] = f"{int(block)}"
            userdict[name]['in_degree_total'] = 0
            userdict[name]['out_degree_total'] = 0
            userdict[name]['N_trends_total'] = 0

print("Making dataframe...")
userdf = pd.DataFrame(userdict).T

userdf = userdf.reset_index()
userdf = userdf.rename(columns={'index':'user_id'})

print("Saving...")
userdf.to_csv(OUTFILE,index=False)            

