# lqg_simulation/plotting/visualize.py
"""
Utilities for visualizing spin networks using NetworkX and Matplotlib.
"""
import networkx as nx
import matplotlib.pyplot as plt
from lqg_simulation.core import SpinNetwork, Link


def plot_spin_network(network: SpinNetwork, title="Spin Network", highlight_links: list[Link] = None):
    """
    Generates a 2D plot of a spin network.

    Args:
        network: The SpinNetwork object to plot.
        title: The title for the plot.
        highlight_links: An optional list of Link objects to draw in a
                         different color to represent a surface.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    G = nx.Graph()
    node_map = {node.name: node for node in network.nodes}
    for node_name in node_map:
        G.add_node(node_name)
    for link in network.links:
        G.add_edge(link.node1.name, link.node2.name, spin=f"j={link.spin_j}")

    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700, ax=ax)

    # Draw edges, with optional highlighting
    highlight_set = set(highlight_links) if highlight_links else set()
    standard_edges = [e for u, v in G.edges() if (link := network.get_link_between(node_map[u], node_map[v])) and link not in highlight_set for e in [(u,v)]]
    highlight_edges = [e for u, v in G.edges() if (link := network.get_link_between(node_map[u], node_map[v])) and link in highlight_set for e in [(u,v)]]

    nx.draw_networkx_edges(G, pos, edgelist=standard_edges, width=1.0, alpha=0.5, edge_color='k', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=2.5, alpha=0.8, edge_color='red', ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'spin')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', ax=ax)

    ax.set_title(title)
    ax.margins(0.1)
    plt.axis("off")
    return fig, ax


def plot_observable_evolution(values, observable_name="Observable", ylabel=None):
    """
    Plot the evolution of an observable (e.g., area, volume) over simulation steps.

    Args:
        values: List of observable values at each step.
        observable_name: Name of the observable (for title).
        ylabel: Label for the y-axis (default: observable_name).

    Returns:
        Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(values)), values, marker='o')
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or observable_name)
    ax.set_title(f"Evolution of {observable_name}")
    ax.grid(True)
    return fig, ax