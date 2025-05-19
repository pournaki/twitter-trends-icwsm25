### ================== SBM CLUSTERING ================== ###
###
### Takes each preprocessed retweet network file from 'output/rtn/'
### and the fdl file with columns ['user_id', 'x', 'y'] from 'output/fdl/'
### and creates a file for each retweet network in 'output/sbm/':
### - ./output/sbm/{filename}_sbm.csv: block structure with columns ['user_id','block','indegree','outdegree']
### - ./output/sbm/sbm_stats.csv: ['trend','N_users','N_links','N_clusters','silhouette_score','N_sbm_runs','computation_time']

### ========= adapt this based on your graph-tool installation =================
import sys
sys.path.append("/opt/local/graphtool/2.63/lib/python3/dist-packages")   ## latest version as of 2024-05-09
# sys.path.append("/opt/local/graphtool/2.59/lib/python3/dist-packages")
# sys.path.append("/opt/local/graphtool/2.45/lib/python3/dist-packages") ## requires deb10
import graph_tool.all as gt
### ===========================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import csv
import datetime as dt

## PARAMETERS
N_sbm_runs = 10
T_silhouette = 0.4
planted_partition_mode = False
outdir = "./output/sbm/"

def infer_block_model(G,
                      fdl_coords,
                      directed=True,
                      hierarchical=False,
                      planted_partition=False,
                      min_blocks=None,
                      max_blocks=None,                      
                      verbose=False):
    """ Infer SBM using Tiago Peixoto's graph-tool
    
    Input:
    graph-tool parameters
    
    Output:
    CSV with original node names and block assignment
    
    """
    
    tic = dt.datetime.now()
        
    GT = G.copy()
    
    if hierarchical == False:
        
        if planted_partition == False:
            
            if min_blocks != None and max_blocks != None:
                ## we need to do this because of a bug in graph-tool...
                state = gt.minimize_blockmodel_dl(GT,
                                                  multilevel_mcmc_args={'B_min':min_blocks,
                                                                        'B_max':max_blocks})
            else:
                state = gt.minimize_blockmodel_dl(GT)

        elif planted_partition == True:

            if min_blocks != None and max_blocks != None:
                state = gt.minimize_blockmodel_dl(GT,
                                                  state=gt.PPBlockState,
                                                  multilevel_mcmc_args={'B_min':min_blocks,
                                                                        'B_max':max_blocks})

            else:
                state = gt.minimize_blockmodel_dl(GT,
                                                  state=gt.PPBlockState)        
                    
        
    if hierarchical == True:
        return("NOT IMPLEMENTED YET")
    
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    blockstates = np.array(list(state.get_blocks()))

    ## export the block structure
    blockdf = pd.DataFrame({'user_id':list(GT.vertex_properties['name']),
                            'block':blockstates+1,
                            'indegree':list(GT.degree_property_map('in')),
                            'outdegree':list(GT.degree_property_map('out'))})

    blockdf = blockdf.set_index('user_id')
    
    toc = dt.datetime.now()
    
    if verbose == True:
        print("NETWORK")
        print(f"N_nodes: {GT.num_vertices()}")
        print(f"N_edges: {GT.num_edges()}")
        print("=======================================")
        print("PARAMETERS:")
        print(f"Directed: {directed}")
        print(f"Hierarchical: {hierarchical}")
        print(f"Planted Partition: {planted_partition}")
        print(f"Min. N_blocks: {min_blocks}")
        print(f"Max. N_blocks: {max_blocks}")
        print("=======================================")
        print("RESULTS")
        print(f"N_blocks: {max(blockstates)+1}")
        print("=======================================")
        print(f"LL=-{state.entropy()}")        
        plot_solution(blockdf)
    
    plotdf = fdl_coords.join(blockdf)

    ## ============================ CAUTION ============================
    ## these two lines should not exist if matlab and graph-tool did the same thing
#     plotdf = plotdf[plotdf['x'].notna()]
#     plotdf = plotdf[plotdf['block'].notna()]
    ## ============================ CAUTION ============================
    
    xy = np.array(plotdf[['x','y']])
    labels = np.array(plotdf['block'])
    
#     print(len(labels))
    
    if len(set(labels)) > 1:
        shsc = silhouette_score(xy,labels)
    else:
        shsc = -2
        
    if verbose == True:
        print(f"Silhouette score = {shsc}")

        print(toc-tic)
        print("=======================================")
        print("=======================================")    
    
    return state,blockdf,shsc


RTN_DIR = "./output/rtn/"
FDL_DIR = "./output/fdl/"
fdl_files = sorted(os.listdir(FDL_DIR))
fdl_files = [i for i in sorted(os.listdir(FDL_DIR)) if "csv" in i]
rtn_files = [i.replace("_fdl.csv","_rtn_pp.csv") for i in fdl_files]

## sanity check
e = []
for f in rtn_files:
    if not os.path.exists(RTN_DIR+f):
        print(f)
        e.append(f)        
        
if len(e) == 0:
    print('Sanity check passed')
    
print(f"{(len(fdl_files))} networks to process.")    

if not os.path.exists(outdir):
    os.mkdir(outdir)

## initialize stats
STAT_FIELDS = ["trend","N_users","N_links","N_clusters","silhouette_score","N_sbm_runs","computation_time"]
statfile = outdir + "sbm_stats.csv"
with open (statfile,"w",encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(STAT_FIELDS)

## load the graph and the coordinates
for idx in tqdm(range(len(fdl_files))):
    
    ## read the coordinates of the force-directed layout computed in matlab
    fdl_coordinates = pd.read_csv(FDL_DIR+fdl_files[idx],
                             dtype={'user id':'str'})
    fdl_coordinates = fdl_coordinates.rename(columns={'user id':'user_id'})
    fdl_coordinates = fdl_coordinates.set_index('user_id')
    
    if len(fdl_coordinates) >= 50:
    
        tic = dt.datetime.now()

        outfile = rtn_files[idx]
        outfile = outdir + outfile.replace("_rtn_pp.csv","_sbm.csv")    
        outfile_sil = outfile.replace("_sbm.csv","_sil.csv")

        # print(rtn_files[idx])

        G = gt.load_graph_from_csv(RTN_DIR+rtn_files[idx],
                                   directed=True,
                                   ecols=(0,1),
                                   skip_first=True
                                   )

        N_users = G.num_vertices()
        N_links = G.num_edges()

        trendname = rtn_files[idx].split("_rtn_pp")[0]

        ## run the block model
        blockdfs = []
        scores = []
        for i in range(N_sbm_runs):
            state,blockdf,shsc = infer_block_model(G,
                                                   fdl_coords=fdl_coordinates,
                                                   directed=True,
                                                   hierarchical=False,
                                                   planted_partition=planted_partition_mode,
                                                   min_blocks=1,
                                                   max_blocks=2,
                                                   verbose=False)
            blockdfs.append(blockdf)
            scores.append(shsc)

        max_shsc = np.max(scores)
        # print("S_max:",max_shsc)
        if max_shsc >= T_silhouette:        
            selected_blockstate = blockdfs[np.argmax(scores)]
            # plot_solution(selected_blockstate,fdl_coordinates)

        else:
            selected_blockstate = blockdfs[0]
            selected_blockstate['block'] = 1
            # plot_solution(selected_blockstate,fdl_coordinates)

        toc = dt.datetime.now()

        computation_time = str(toc-tic)

        ## write the result
        selected_blockstate.reset_index().to_csv(outfile,index=False)

        ## write the stats
        N_blocks = len(set(selected_blockstate['block']))
        with open (statfile,"a",encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([trendname,N_users,N_links,N_blocks,max_shsc,N_sbm_runs,computation_time])

    #     print("\n==============================================\n")

