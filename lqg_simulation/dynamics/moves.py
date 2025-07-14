# lqg_simulation/dynamics/moves.py
"""
Basic dynamic moves for spin networks: change spin, add/remove node/link.
"""

def change_spin(link, new_spin):
    """Change the spin of a link."""
    link.spin_j = new_spin

def add_node(network, name=None):
    """Add a node to the network."""
    return network.add_node(name)

def remove_node(network, node):
    """Remove a node and all its links from the network."""
    network.nodes.remove(node)
    network.links = [l for l in network.links if l.node1 != node and l.node2 != node]

def add_link(network, node1, node2, spin_j=1, link_name=None):
    """Add a link between two nodes."""
    return network.add_link(node1, node2, spin_j, link_name)

def remove_link(network, link):
    """Remove a link from the network."""
    network.links.remove(link)
