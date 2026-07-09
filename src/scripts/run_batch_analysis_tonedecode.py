#!/usr/bin/env python3
"""
Batch processing script for arousal network analysis across multiple sessions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pickle
import warnings
import subprocess

warnings.filterwarnings("ignore")

# Setup paths
repoDir = Path(__file__).resolve().parent.parent.parent
# print(repoDir)
configDir = repoDir / "config"
sys.path.append(str(configDir))

from settings import DATA_PATH, CODE_PATH

sys.path.append(str(CODE_PATH["2-photon"]))


class ArousalNetworkBatchProcessor:
    def __init__(self, base_data_path, results_dir):
        self.base_data_path = Path(base_data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Store results across sessions
        self.batch_results = {}

    def find_sessions(self):
        """Find all sessions with required files."""
        sessions = []
        for session_dir in self.base_data_path.iterdir():
            if session_dir.is_dir():
                session_pkl = session_dir / "sessionProcessDF.pkl"
                network_pkl = (
                    session_dir / "neural_pupil_network_aligned_temporally.pkl"
                )

                if session_pkl.exists() and network_pkl.exists():
                    sessions.append(session_dir)

        return sessions

    def run_neural_pupil_processing(self, session_path):
        """Run the existing neural pupil network processing script."""
        print(f"Running neural pupil network processing for {session_path.name}...")

        try:
            cmd = [
                "python",
                "stimuli_phys_network_align_for_batch.py",
                "--session_path",
                str(session_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(CODE_PATH["2-photon"], "scripts")),
            )  ## change this to scripts if needed when I move this to scripts
            if result.returncode != 0:
                print(f"Error processing {session_path.name}: {result.stderr}")
                return False
            else:
                print(f"Successfully processed {session_path.name}")
                return True

        except Exception as e:
            print(f"Error running processing script for {session_path.name}: {e}")
            return False

    def analyze_session(self, session_path):
        """Run the arousal network analysis for a single session."""
        session_name = session_path.name
        print(f"\n{'=' * 60}")
        print(f"Analyzing session: {session_name}")
        print(f"{'=' * 60}")

        try:
            # Load data
            # session_df = pd.read_pickle(session_path / "sessionProcessDF.pkl")

            with open(
                session_path / "neural_pupil_network_aligned_temporally.pkl", "rb"
            ) as f:
                neural_pupil_network = pickle.load(f)

            # Extract variables
            pupil_data = neural_pupil_network["pupil"]
            neural_data = neural_pupil_network["spikes"]
            stimuli_data = neural_pupil_network["stimuli"]
            network_data = neural_pupil_network["network"]

            # Run arousal partitioning analysis
            results = self.run_arousal_analysis(pupil_data, network_data, session_name)

            # Save session results
            session_results_dir = self.results_dir / session_name
            session_results_dir.mkdir(exist_ok=True)

            # Save detailed results
            with open(session_results_dir / "arousal_analysis_results.pkl", "wb") as f:
                pickle.dump(results, f)

            # Save metrics CSV
            if "metrics_df" in results:
                results["metrics_df"].to_csv(session_results_dir / "graph_metrics.csv")

            # Store in batch results
            self.batch_results[session_name] = results

            print(f"Successfully analyzed {session_name}")
            return results

        except Exception as e:
            print(f"Error analyzing {session_name}: {e}")
            self.batch_results[session_name] = {"error": str(e)}
            return None

    def run_arousal_analysis(self, pupil_data, network_data, session_name):
        """Core arousal analysis - extracted from your notebook."""

        # Extract pupil diameter and time data
        pupil_diameter = pupil_data["diameter"]
        pupil_times = pupil_data["time_sec"]
        agc_network = network_data["adjacency_matrix"]
        network_times = network_data["time_bins_raw"]

        n_cells, _, n_time_bins = agc_network.shape

        print(f"Pupil data shape: {pupil_diameter.shape}")
        print(f"AGC network shape: {agc_network.shape}")

        # Remove NaN values for partitioning
        valid_pupil_mask = ~np.isnan(pupil_diameter)
        valid_pupil_diameter = pupil_diameter[valid_pupil_mask]
        valid_pupil_times = pupil_times[valid_pupil_mask]

        print(
            f"Valid pupil samples: {len(valid_pupil_diameter)} / {len(pupil_diameter)}"
        )

        # Calculate percentiles for partitioning
        low_threshold = np.percentile(valid_pupil_diameter, 40)
        high_threshold = np.percentile(valid_pupil_diameter, 80)

        print(f"Arousal thresholds: Low={low_threshold:.3f}, High={high_threshold:.3f}")

        # Create arousal masks
        low_arousal_mask = valid_pupil_diameter < low_threshold
        medium_arousal_mask = (valid_pupil_diameter >= low_threshold) & (
            valid_pupil_diameter < high_threshold
        )
        high_arousal_mask = valid_pupil_diameter >= high_threshold

        # Get time indices for each arousal level
        low_arousal_times = valid_pupil_times[low_arousal_mask]
        medium_arousal_times = valid_pupil_times[medium_arousal_mask]
        high_arousal_times = valid_pupil_times[high_arousal_mask]

        print(f"Arousal distribution:")
        print(
            f"  Low: {len(low_arousal_times)} samples ({len(low_arousal_times) / len(valid_pupil_diameter) * 100:.1f}%)"
        )
        print(
            f"  Medium: {len(medium_arousal_times)} samples ({len(medium_arousal_times) / len(valid_pupil_diameter) * 100:.1f}%)"
        )
        print(
            f"  High: {len(high_arousal_times)} samples ({len(high_arousal_times) / len(valid_pupil_diameter) * 100:.1f}%)"
        )

        # Find network indices
        def find_network_indices(arousal_times, network_times):
            network_indices = []
            for t in arousal_times:
                idx = np.argmin(np.abs(network_times - t))
                network_indices.append(idx)
            return np.array(network_indices)

        low_arousal_network_idx = find_network_indices(low_arousal_times, network_times)
        medium_arousal_network_idx = find_network_indices(
            medium_arousal_times, network_times
        )
        high_arousal_network_idx = find_network_indices(
            high_arousal_times, network_times
        )

        # Remove out-of-bounds indices
        low_arousal_network_idx = low_arousal_network_idx[
            low_arousal_network_idx < agc_network.shape[2]
        ]
        medium_arousal_network_idx = medium_arousal_network_idx[
            medium_arousal_network_idx < agc_network.shape[2]
        ]
        high_arousal_network_idx = high_arousal_network_idx[
            high_arousal_network_idx < agc_network.shape[2]
        ]

        # Extract network data
        low_arousal_networks = agc_network[:, :, low_arousal_network_idx]
        medium_arousal_networks = agc_network[:, :, medium_arousal_network_idx]
        high_arousal_networks = agc_network[:, :, high_arousal_network_idx]

        # Max project networks
        def max_project_to_binary(network_matrices, threshold=0):
            max_projected = np.max(network_matrices, axis=2)
            binary_matrix = (max_projected > threshold).astype(int)
            return max_projected, binary_matrix

        print("Creating max-projected binary network matrices...")

        low_max_proj, low_binary = max_project_to_binary(low_arousal_networks)
        medium_max_proj, medium_binary = max_project_to_binary(medium_arousal_networks)
        high_max_proj, high_binary = max_project_to_binary(high_arousal_networks)

        max_projected_networks = {
            "low": {
                "max_projected": low_max_proj,
                "binary_matrix": low_binary,
                "n_connections": np.sum(low_binary),
                "connection_density": np.sum(low_binary) / low_binary.size,
            },
            "medium": {
                "max_projected": medium_max_proj,
                "binary_matrix": medium_binary,
                "n_connections": np.sum(medium_binary),
                "connection_density": np.sum(medium_binary) / medium_binary.size,
            },
            "high": {
                "max_projected": high_max_proj,
                "binary_matrix": high_binary,
                "n_connections": np.sum(high_binary),
                "connection_density": np.sum(high_binary) / high_binary.size,
            },
        }

        # Compute graph metrics
        graph_metrics = {}
        for arousal_level in ["low", "medium", "high"]:
            binary_matrix = max_projected_networks[arousal_level]["binary_matrix"]
            metrics = self.compute_graph_metrics(binary_matrix, arousal_level)
            graph_metrics[arousal_level] = metrics

        # Create DataFrame
        metrics_df = pd.DataFrame(graph_metrics).T

        # Find hubs
        hubs = {}
        for arousal_level in ["low", "medium", "high"]:
            binary_matrix = max_projected_networks[arousal_level]["binary_matrix"]
            G = nx.from_numpy_array(binary_matrix.T, create_using=nx.DiGraph)
            G.remove_edges_from(nx.selfloop_edges(G))
            degrees = G.out_degree()
            sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
            hubs[arousal_level] = sorted_degrees[:5]

        return {
            "session_name": session_name,
            "thresholds": {"low": low_threshold, "high": high_threshold},
            "max_projected_networks": max_projected_networks,
            "graph_metrics": graph_metrics,
            "metrics_df": metrics_df,
            "hubs": hubs,
            "n_cells": n_cells,
        }

    def compute_graph_metrics(self, binary_matrix, arousal_level):
        """Compute graph theory metrics - from your notebook."""
        G = nx.from_numpy_array(binary_matrix.T, create_using=nx.DiGraph)
        G.remove_edges_from(nx.selfloop_edges(G))

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)

        G_undirected = G.to_undirected()

        metrics = {
            "arousal_level": arousal_level,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "is_connected": nx.is_strongly_connected(G),
            "n_components": nx.number_strongly_connected_components(G),
        }

        # Clustering metrics
        try:
            metrics["avg_clustering"] = nx.average_clustering(G_undirected)
            metrics["transitivity"] = nx.transitivity(G_undirected)
        except:
            metrics["avg_clustering"] = 0
            metrics["transitivity"] = 0

        # Modularity
        try:
            if G_undirected.number_of_edges() > 0:
                communities = nx.community.louvain_communities(G_undirected, seed=79)
                metrics["modularity"] = nx.algorithms.community.modularity(
                    G_undirected, communities
                )
                metrics["n_communities"] = len(communities)
            else:
                metrics["modularity"] = 0
                metrics["n_communities"] = 0
        except:
            metrics["modularity"] = 0
            metrics["n_communities"] = 0

        return metrics

    def create_summary_report(self):
        """Create a summary report across all sessions."""
        print(f"\n{'=' * 60}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'=' * 60}")

        successful_sessions = [
            name
            for name, results in self.batch_results.items()
            if "error" not in results
        ]
        failed_sessions = [
            name for name, results in self.batch_results.items() if "error" in results
        ]

        print(f"Successful sessions: {len(successful_sessions)}")
        print(f"Failed sessions: {len(failed_sessions)}")

        if failed_sessions:
            print("\nFailed sessions:")
            for session in failed_sessions:
                print(f"  - {session}: {self.batch_results[session]['error']}")

        if successful_sessions:
            # Create comparison DataFrame
            comparison_data = []
            for session_name in successful_sessions:
                results = self.batch_results[session_name]
                for arousal_level in ["low", "medium", "high"]:
                    metrics = results["graph_metrics"][arousal_level]
                    row = {
                        "session": session_name,
                        "arousal_level": arousal_level,
                        "density": metrics["density"],
                        "avg_clustering": metrics["avg_clustering"],
                        "modularity": metrics["modularity"],
                        "n_communities": metrics["n_communities"],
                    }
                    comparison_data.append(row)

            comparison_df = pd.DataFrame(comparison_data)

            # Save comparison
            comparison_df.to_csv(self.results_dir / "batch_comparison.csv", index=False)

            # Create summary plots
            self.create_summary_plots(comparison_df)

            print(f"\nSummary saved to: {self.results_dir}")

        return comparison_df if successful_sessions else None

    def create_summary_plots(self, comparison_df):
        """Create summary plots across sessions."""
        metrics_to_plot = ["density", "avg_clustering", "modularity"]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))

        for i, metric in enumerate(metrics_to_plot):
            sns.boxplot(data=comparison_df, x="arousal_level", y=metric, ax=axes[i])
            axes[i].set_title(f"{metric.replace('_', ' ').title()} Across Sessions")
            axes[i].set_xlabel("Arousal Level")
            axes[i].set_ylabel(metric.replace("_", " ").title())

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "batch_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def run_batch_processing(self, sessions=False, run_preprocessing=True):
        """Run the complete batch processing pipeline."""
        print("Starting Batch Arousal Network Analysis")
        print(f"Base data path: {self.base_data_path}")
        print(f"Results directory: {self.results_dir}")

        # Find sessions
        if not sessions:
            sessions = self.find_sessions()
        else:
            sessions = [self.base_data_path / s for s in sessions]
        print(f"Found {len(sessions)} sessions to process")

        for session_path in sessions:
            session_name = session_path.name

            # Step 1: Run neural pupil network processing if needed
            if run_preprocessing:
                network_file = (
                    session_path / "neural_pupil_network_aligned_temporally.pkl"
                )
                if not network_file.exists():
                    success = self.run_neural_pupil_processing(session_path)
                    if not success:
                        print(f"Skipping {session_name} due to preprocessing failure")
                        continue

            # Step 2: Run arousal analysis
            self.analyze_session(session_path)

        # Step 3: Create summary report
        summary_df = self.create_summary_report()

        return summary_df


def main():
    """Main execution function."""

    session_names = [
        "2022-05-09_13-52-25_mouse-0951",
        "2022-05-09_14-25-05_mouse-0951",
    ]

    # Setup paths
    base_data_path = Path(DATA_PATH["toneDecode"])
    results_dir = Path(DATA_PATH["toneDecode"], "arousal_analysis_results")
    # Create processor obj
    processor = ArousalNetworkBatchProcessor(base_data_path, results_dir)

    # Run batch processing
    summary_df = processor.run_batch_processing(
        run_preprocessing=True, sessions=session_names
    )

    if summary_df is not None:
        print("\nBatch processing completed successfully!")
        print(f"Results saved to: {results_dir}")

        # Print summary statistics
        print("\nSummary statistics:")
        for metric in ["density", "avg_clustering", "modularity"]:
            print(f"\n{metric.upper()}:")
            summary_stats = summary_df.groupby("arousal_level")[metric].agg(
                ["mean", "std"]
            )
            print(summary_stats)
    else:
        print("\nBatch processing failed")


if __name__ == "__main__":
    main()
