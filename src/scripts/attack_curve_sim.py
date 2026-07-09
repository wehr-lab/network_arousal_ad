import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nctpy.metrics import ave_control


# Functions to compute metrics
def compute_rich_club(G, k=20):
    rich_nodes = [n for n in G.nodes if G.degree[n] > k]
    subgraph = G.subgraph(rich_nodes)
    possible_edges = len(rich_nodes) * (len(rich_nodes) - 1) / 2
    return subgraph.number_of_edges() / possible_edges if possible_edges else 0


def compute_modularity(G):
    """Compute modularity using Louvain community detection"""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0
    try:
        # Get the largest connected component if graph is disconnected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        communities = nx.community.louvain_communities(G)
        return nx.community.modularity(G, communities)
    except:
        return 0


def compute_global_efficiency(G):
    """Compute global efficiency of the network"""
    if G.number_of_nodes() == 0:
        return 0
    try:
        return nx.global_efficiency(G)
    except Exception as e:
        print(f"Global efficiency error: {e}")
        return 0


def compute_small_worldness(G):
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    C = nx.average_clustering(G)
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError as e:
        print(f"Average shortest path length error: {e}")
        return 0
    deg_seq = [d for _, d in G.degree()]
    G_rand = nx.configuration_model(deg_seq)
    G_rand = nx.Graph(G_rand)
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    if not nx.is_connected(G_rand):
        largest_cc = max(nx.connected_components(G_rand), key=len)
        G_rand = G_rand.subgraph(largest_cc).copy()
    C_rand = nx.average_clustering(G_rand)
    try:
        L_rand = nx.average_shortest_path_length(G_rand)
    except nx.NetworkXError:
        return 0
    return (C / C_rand) / (L / L_rand) if C_rand and L_rand else 0


def compute_average_controllability(G, system="discrete"):
    """Compute average controllability of the network"""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0
    try:
        # Get the adjacency matrix
        A = nx.to_numpy_array(G)

        # Normalize the adjacency matrix (required for controllability)
        # Use the largest eigenvalue for normalization
        eigenvalues = np.linalg.eigvals(A)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            A_normalized = A / (max_eigenvalue + 0.01)
        else:
            A_normalized = A

        # Compute average controllability for all nodes
        avg_ctrl = ave_control(A_normalized, system=system)

        # Return the mean average controllability across all nodes
        result = np.mean(avg_ctrl)
        return result
    except Exception as e:
        print(f"Controllability error: {e}")
        return 0


def create_sbm_graph(n_nodes, n_communities, p_in, p_out):
    """
    Create a Stochastic Block Model graph.

    Parameters:
    -----------
    n_nodes : int
        Total number of nodes
    n_communities : int
        Number of communities/blocks
    p_in : float
        Probability of edge within a community
    p_out : float
        Probability of edge between communities

    Returns:
    --------
    G : networkx.Graph
        Generated SBM graph
    """
    nodes_per_community = n_nodes // n_communities
    sizes = [nodes_per_community] * n_communities
    # Add remaining nodes to the last community
    sizes[-1] += n_nodes - sum(sizes)

    # Create probability matrix
    # High probability within blocks, low probability between blocks
    probs = [[p_out for _ in range(n_communities)] for _ in range(n_communities)]
    for i in range(n_communities):
        probs[i][i] = p_in

    # Generate SBM graph
    G = nx.stochastic_block_model(sizes, probs, seed=np.random.randint(0, 10000))

    return G


def perturb_graph(adj_matrix, p_remove=0.1, p_add=0.1):
    """
    Add variance to adjacency matrix by randomly removing/adding edges.

    Parameters:
    -----------
    adj_matrix : numpy array
        Binary adjacency matrix
    p_remove : float
        Probability of removing an existing edge
    p_add : float
        Probability of adding a new edge

    Returns:
    --------
    perturbed_matrix : numpy array
        Perturbed adjacency matrix
    """
    perturbed = adj_matrix.copy()
    n = perturbed.shape[0]

    # Remove some existing edges
    existing_edges = np.argwhere(np.triu(perturbed, k=1) == 1)
    n_remove = int(len(existing_edges) * p_remove)
    if n_remove > 0:
        remove_idx = np.random.choice(len(existing_edges), n_remove, replace=False)
        for idx in remove_idx:
            i, j = existing_edges[idx]
            perturbed[i, j] = 0
            perturbed[j, i] = 0

    # Add some new edges
    non_edges = np.argwhere(np.triu(perturbed, k=1) == 0)
    n_add = int(len(non_edges) * p_add)
    if n_add > 0:
        add_idx = np.random.choice(len(non_edges), n_add, replace=False)
        for idx in add_idx:
            i, j = non_edges[idx]
            perturbed[i, j] = 1
            perturbed[j, i] = 1

    return perturbed


# Simulation settings
# n_nodes = 500
n_subjects = 20
attack_steps = 15

# Store metric trajectories
rc_rich, rc_rand = [], []
sw_rich, sw_rand = [], []
mod_rich, mod_rand = [], []
ge_rich, ge_rand = [], []
ac_rich, ac_rand = [], []

bin_mat = np.load(
    "/Users/praveslamichhane/Desktop/network_compute_experiment/data/processed/bin_mat.npy",
    allow_pickle=True,
)

# # Get parameters from the actual data for SBM
# n_nodes = bin_mat.shape[0]
# n_communities = 4  # Based on your clustering analysis

# # Calculate edge density from the real data to calibrate p_in and p_out
# edge_density = np.sum(bin_mat) / (n_nodes * (n_nodes - 1))
# p_in = min(edge_density * 3, 0.8)  # Higher within-community connectivity, capped at 0.8
# p_out = edge_density * 0.5  # Lower between-community connectivity

# print(f"Number of nodes: {n_nodes}")
# print(f"Number of communities: {n_communities}")
# print(f"Edge density: {edge_density:.4f}")
# print(f"Within-community edge probability (p_in): {p_in:.4f}")
# print(f"Between-community edge probability (p_out): {p_out:.4f}")


# Simulation loop
for _ in range(n_subjects):
    perturbed_mat = perturb_graph(bin_mat, p_remove=0.1, p_add=0.1)
    G_base = nx.from_numpy_array(perturbed_mat)

    G1 = G_base.copy()  ## for rich-club attack
    G2 = G_base.copy()  ## for random attack
    r1, r2 = [], []  ## rich-club coefficients but for 1 subject
    s1, s2 = [], []  ## small-worldness but for 1 subject
    m1, m2 = [], []  ## modularity but for 1 subject
    g1, g2 = [], []  ## global efficiency but for 1 subject
    a1, a2 = [], []  ## average controllability but for 1 subject

    for _ in range(attack_steps):
        r1.append(compute_rich_club(G1))
        s1.append(compute_small_worldness(G1))
        m1.append(compute_modularity(G1))  # Add modularity computation
        g1.append(compute_global_efficiency(G1))  # Add global efficiency computation
        a1.append(compute_average_controllability(G1))
        r2.append(compute_rich_club(G2))
        s2.append(compute_small_worldness(G2))
        m2.append(compute_modularity(G2))  # Add modularity computation
        g2.append(compute_global_efficiency(G2))  # Add global efficiency computation
        a2.append(compute_average_controllability(G2))

        # Remove a random node from the rich-club (top 20% by degree) in each step of the iteration
        degrees = dict(G1.degree())
        degree_threshold = np.percentile(list(degrees.values()), 80)
        rich_nodes = [n for n, d in degrees.items() if d >= degree_threshold]
        if rich_nodes:
            G1.remove_node(np.random.choice(rich_nodes))

        G2.remove_node(np.random.choice(list(G2.nodes)))

    rc_rich.append(r1)
    rc_rand.append(r2)
    sw_rich.append(s1)
    sw_rand.append(s2)
    mod_rich.append(m1)
    mod_rand.append(m2)
    ge_rich.append(g1)
    ge_rand.append(g2)
    ac_rich.append(a1)
    ac_rand.append(a2)


# Prepare DataFrames
def aggregate_results(trajectories, label):
    df = pd.DataFrame(trajectories).T
    df["Step"] = df.index
    df = df.melt(id_vars="Step", var_name="Subject", value_name="Value")
    df["AttackType"] = label
    return df


df_rc = pd.concat(
    [aggregate_results(rc_rich, "Rich-Club"), aggregate_results(rc_rand, "Random")]
)
df_rc["Metric"] = "Rich-Club Coefficient"

df_sw = pd.concat(
    [aggregate_results(sw_rich, "Rich-Club"), aggregate_results(sw_rand, "Random")]
)
df_sw["Metric"] = "Small-Worldness"

df_mod = pd.concat(
    [aggregate_results(mod_rich, "Rich-Club"), aggregate_results(mod_rand, "Random")]
)
df_mod["Metric"] = "Modularity"

df_ge = pd.concat(
    [aggregate_results(ge_rich, "Rich-Club"), aggregate_results(ge_rand, "Random")]
)
df_ge["Metric"] = "Global Efficiency"

df_ac = pd.concat(
    [aggregate_results(ac_rich, "Rich-Club"), aggregate_results(ac_rand, "Random")]
)
df_ac["Metric"] = "Average Controllability"


## concat only what we need for plotting
df_all = pd.concat([df_mod, df_ge], ignore_index=True)

# Plotting
sns.set(style="whitegrid", font_scale=1.2)
metrics = df_all["Metric"].unique()
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

for ax, metric in zip(axes, metrics):
    sns.lineplot(
        data=df_all[df_all["Metric"] == metric],
        x="Step",
        y="Value",
        hue="AttackType",
        errorbar=("ci", 95),
        palette="Set1",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(metric)
    ax.set_xlabel("Nodes Removed")
    ax.set_ylabel("Value")
    ax.legend(title="Attack Type")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="y", rotation=45)

plt.suptitle("Attack Impact on Network Metrics (95% Confidence Interval)", fontsize=16)
plt.tight_layout()  ## rect=[0, 0, 1, 0.95]
# plt.savefig(
#     "/Users/praveslamichhane/Desktop/attack_curves.png", dpi=150, bbox_inches="tight"
# )
# print("\nPlot saved to Desktop/attack_curves.png")
plt.show()
