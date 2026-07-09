## import libraries 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
import sys 

## add path to custom libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ephys_analysis 
import behavior_analysis

## define path variables -> note to make these paths genaralizable by doing sys.argv[]
sessionPaths = ["/Volumes/projects/2P5XFAD/JarascopeData/wehr3204/01-17-25-001/suite2p/plane0", "/Volumes/projects/2P5XFAD/JarascopeData/wehr3204/01-22-25-001/suite2p/plane0", "/Volumes/projects/2P5XFAD/JarascopeData/wehr3204/01-30-25-001/suite2p/plane0"]

# sessionPaths = ["/Volumes/projects/2P5XFAD/JarascopeData/wehr3204/01-22-25-002/suite2p/plane0", "/Volumes/projects/2P5XFAD/JarascopeData/wehr3204/01-30-25-002/suite2p/plane0"]

dFFs = {} ## this will store the dFF values for each session

for session in sessionPaths:
    ## load data 
    F, Fneu, spks, stat, ops, iscell = ephys_analysis.load2P(session) 

    print("Data loaded for session: ", session)

    ## NOTE: data loading is taking a long time. This is because the data is being loaded from a network drive. This will be the same when we work on talapas. Maybe there is a better way? 

    indicesCells = ephys_analysis.getIndices(iscell, prob=0.8)
    fCells = ephys_analysis.extractCells(F, indicesCells)
    fneuCells = ephys_analysis.extractCells(Fneu, indicesCells)

    ## neuropil correction
    fCorrected = ephys_analysis.correct_neuropil(fCells, fneuCells, npil_coeff=0.7)

    ## remove negative values from the corrected fluorescence values
    fCorrectedShifted = ephys_analysis.absFluoresce(fCorrected)

    ## another plot to visualize the time series of the corrected fluorescence values and to select a threshold
    # plt.plot(fCorrectedShifted[0, :])

    ## draw a horizontal line at the 10th percentile
    # plt.axhline(np.percentile(fCorrectedShifted[0, :], 20), color="red")
    # plt.show() 

    ## compute dF/F using a sliding window
    dFF = ephys_analysis.computeDFFSlide(fCorrectedShifted, window=200, percentile=20) ## note that the window size and the percentile are parameters are totally arbitrary for now 
    print("Sliding window dF/F computed for session: ", session)

    # ## save data for further processing 
    # np.save(os.path.join(DATA_PATH_SAVE, "dF_F_100824_sliding_f_nought.npy"), dFF)
    dFFConvolve = ephys_analysis.convolve_spikes(dFF, sigma=10)
    dFFs[session] = dFFConvolve

# ## save the dFF values for each session as .npy files and load them -> faster
savePath = sessionPaths[0].split("/")[:-3]
savePath = "/".join(savePath) + "/"
np.savez(savePath + "dFFs_high_cell_prob_tones.npy", dFFs)

# ## load the dFF values for each session
# savedFile = np.load(savePath + "dFFs_high_cell_prob_tones.npy.npz", allow_pickle=True)
# dFFs = savedFile["arr_0"].item()

fcMatrix = {}
fcMatrixBin = {}
for session, dFF in dFFs.items():
    ## compute the correlation matrix
    fcMatrix[session] = np.corrcoef(dFF)
    print("Functional connectivity matrix computed for session: ", session)

    ## compute the 20th percentile of the correlation matrix
    matToProcess = fcMatrix[session]
    upperTriangular = matToProcess[np.triu_indices_from(matToProcess, k=1)] 
    print(f"Max and min values for {session} are: {np.max(upperTriangular)}, {np.min(upperTriangular)}")
    # thresholdPercentile = np.percentile(upperTriangular.flatten(), 40)
    absThreshold = 0.1 
    print(f"Threshold percentile for session {session}: ", absThreshold)
    ## realized that for the 2nd matrix, 0.1 is really high. For example, only 5% of values are retained when the threshold is set to 0.1.
    # sum(upperTriangular > absThreshold)/len(upperTriangular) ## this is the percentage of values that are retained when the threshold is set to 0.1 
    ## binarize the correlation matrix at the threshold percentile 
    fcMatrixBin[session] = ephys_analysis.binarize_matrix(np.abs(matToProcess), threshold=absThreshold)
    ## remove autocorrelations
    np.fill_diagonal(fcMatrix[session], 0)
    np.fill_diagonal(fcMatrixBin[session], 0)

## cluster the non binarized matrix also
for i, (session, fc) in enumerate(fcMatrix.items()):
    g = sns.clustermap(fc, method="ward", cmap="coolwarm", square=True,  xticklabels=False, yticklabels=False, cbar=True, dendrogram_ratio=(0.1, 0.1), col_cluster=True, row_cluster=True, vmin=-0.3, vmax=0.6)
    # if g.cax is not None: ## decided to include the colorbar
    #     g.cax.set_visible(False)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_title(f"Clustering of unbinarized correlation matrix for {session.split('/')[-3]}, Cells={fc.shape[0]}", y=1.05, fontsize=15, fontweight='bold')
    plt.show()

## plot the functional connectivity matrices
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (session, fc) in enumerate(fcMatrixBin.items()):
    g = sns.clustermap(fc, method="ward", cmap="coolwarm", square=True,  xticklabels=False, yticklabels=False, cbar=True, dendrogram_ratio=(0.1, 0.1), col_cluster=True, row_cluster=True)
    # if g.cax is not None: ## decided to include the colorbar
    #     g.cax.set_visible(False)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_title(f"Clustering of binarized correlation matrix for {session.split('/')[-3]}, Cells={fc.shape[0]}", y=1.05, fontsize=15, fontweight='bold')
    plt.show()



## now stuff about the graph theory analysis -> I am using the igraph library for this

## create a graph object from the functional connectivity matrix
graphs = [ig.Graph.Adjacency(fcMatrixBin[session], mode='undirected') for session in sessionPaths]

graphs = [graph.simplify(loops=True, multiple=False) for graph in graphs]


graphMatrics = {}

for i, graph in enumerate(graphs):
    ## compute the degree of each node
    storeDict = {}
    storeDict["degree"] = graph.degree()
    ## compute the clustering coefficient of each node
    storeDict["clustering"] = graph.transitivity_local_undirected()
    ## compute the betweenness centrality of each node
    storeDict["betweenness"] = graph.betweenness()
    ## compute the eigenvector centrality of each node
    storeDict["eigenvector"] = graph.eigenvector_centrality()
    storeDict["path_length"] = graph.average_path_length()
    graphMatrics[i] = storeDict


## plot all the graph metrics
degrees = [np.nan_to_num(np.array(graphMatrics[i]["degree"])) for i in range(len(graphMatrics))]

# ## normalize the degree values according to the number of nodes in the graph
degreesNorm = [degree/(graphs[i].vcount()) for i, degree in enumerate(degrees)] ## think this shoud work 

clusterings = [np.nan_to_num(np.array(graphMatrics[i]["clustering"])) for i in range(len(graphMatrics))]
pathLengths = [np.nan_to_num(np.array(graphMatrics[i]["path_length"])) for i in range(len(graphMatrics))]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].boxplot(degreesNorm)
axs[0].set_xticklabels(["Session 1", "Session 2", "Session 3"]) 
axs[0].set_title("Degree Normalized")
axs[1].boxplot(clusterings)
axs[1].set_xticklabels(["Session 1", "Session 2", "Session 3"]) 
axs[1].set_title("Clustering")
axs[2].bar([1, 2, 3], pathLengths, tick_label=["Session 1", "Session 2", "Session 3"])
# for i, v in enumerate(pathLengths):
#     axs[2].text(i, v + 0.5, f"{v: 2f}", ha='center', fontsize=5) ## did not work properly 
axs[2].set_title("Average Path Length")
fig.suptitle("Graph Metrics for the recording sessions with 0.1 correlation threshold", fontsize=15, fontweight='bold')
plt.show()


