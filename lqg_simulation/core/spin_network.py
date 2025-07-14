# lqg_simulation/core/spin_network.py

import uuid

class Node:
    """
    Represents a node in a spin network.
    """
    def __init__(self, node_id=None, name: str = ""):
        self.id = node_id if node_id is not None else uuid.uuid4()
        self.name = name if name else str(self.id)[:8] # Short user-friendly name

    def __repr__(self):
        return f"Node(id={self.name})"

class Link:
    """
    Represents a link (edge) in a spin network, connecting two nodes
    and carrying a spin quantum number 'j'.
    Spin 'j' is typically a half-integer (0, 1/2, 1, 3/2, ...).
    """
    def __init__(self, node1: Node, node2: Node, spin_j: float, link_id=None, name: str = ""):
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise ValueError("Links must connect two Node objects.")
        if node1 == node2:
            raise ValueError("Self-loops are not allowed in this simple model yet.") # Or handle as needed

        # Check if spin_j is a non-negative integer or half-integer
        is_valid_spin = False
        doubled_spin = 2 * spin_j
        if hasattr(doubled_spin, 'is_integer'): # Check for Sympy object
            if doubled_spin.is_integer and spin_j >= 0:
                is_valid_spin = True
        elif isinstance(spin_j, (int, float)): # Check for Python int/float
            if (doubled_spin % 1 == 0) and spin_j >= 0:
                is_valid_spin = True

        if not is_valid_spin:
            raise ValueError(f"Spin j must be a non-negative integer or half-integer. Got {spin_j}")

        self.id = link_id if link_id is not None else uuid.uuid4()
        self.name = name if name else str(self.id)[:8] # Short user-friendly name
        # Ensure consistent ordering for undirected graph representation if needed later
        self.nodes = tuple(sorted((node1, node2), key=lambda n: n.id.int))
        self.spin_j = spin_j

    @property
    def node1(self):
        return self.nodes[0]

    @property
    def node2(self):
        return self.nodes[1]

    def __repr__(self):
        return f"Link(name={self.name}, nodes=({self.node1.name}, {self.node2.name}), j={self.spin_j})"

    def __eq__(self, other):
        if not isinstance(other, Link):
            return False
        return self.nodes == other.nodes and self.spin_j == other.spin_j

    def __hash__(self):
        return hash((self.nodes, self.spin_j))


class SpinNetwork:
    """
    Represents a spin network as a collection of nodes and links.
    This is essentially a graph where links have spin quantum numbers.
    """
    def __init__(self):
        self.nodes: set[Node] = set()
        self.links: set[Link] = set()
        self._adj: dict[Node, list[Link]] = {} # Adjacency list for quick lookups

    def add_node(self, node: Node = None, node_name: str = "") -> Node:
        if node is None:
            node = Node(name=node_name)
        elif not isinstance(node, Node):
            raise ValueError("Can only add Node objects to the network.")

        if node not in self.nodes:
            self.nodes.add(node)
            self._adj[node] = []
        return node

    def add_link(self, node1: Node, node2: Node, spin_j: float, link_name: str = "") -> Link:
        if node1 not in self.nodes:
            self.add_node(node1)
        if node2 not in self.nodes:
            self.add_node(node2)

        link = Link(node1, node2, spin_j, name=link_name)

        # Avoid duplicate links (same nodes, same spin) - though Link hash/eq handles this for the set
        # More complex logic might be needed if multiple links between same nodes with different spins are allowed
        # or if links are directed. For now, assuming simple undirected graph with unique (nodes, spin) links.
        if link in self.links:
            # Find existing link to return
            for existing_link in self.links:
                if existing_link == link: # Relies on Link.__eq__
                    return existing_link

        self.links.add(link)
        self._adj[node1].append(link)
        self._adj[node2].append(link)
        return link

    def get_node_by_name(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_links_for_node(self, node: Node) -> list[Link]:
        if node not in self.nodes:
            raise ValueError(f"Node {node.name} not in the network.")
        return self._adj.get(node, [])

    def get_neighbors(self, node: Node) -> list[Node]:
        if node not in self.nodes:
            raise ValueError(f"Node {node.name} not in the network.")

        neighbors = []
        for link in self._adj.get(node, []):
            if link.node1 == node:
                neighbors.append(link.node2)
            else:
                neighbors.append(link.node1)
        return neighbors

    def get_link_between(self, node1: Node, node2: Node):
        """Return the link between node1 and node2, or None if not present."""
        for link in self.links:
            if (link.node1 == node1 and link.node2 == node2) or (link.node1 == node2 and link.node2 == node1):
                return link
        return None

    def __repr__(self):
        return f"SpinNetwork(nodes={len(self.nodes)}, links={len(self.links)})"

    def display(self):
        print("Nodes:")
        for node in sorted(list(self.nodes), key=lambda n: n.name):
            print(f"  {node}")
            # For more detail, list connected links:
            # connected_links = [f"{l.name}(j={l.spin_j}, to={l.node2.name if l.node1==node else l.node1.name})" for l in self.get_links_for_node(node)]
            # print(f"    Links: {', '.join(connected_links) if connected_links else 'None'}")


        print("\nLinks:")
        for link in sorted(list(self.links), key=lambda l: l.name):
            print(f"  {link}")

if __name__ == '__main__':
    # Example Usage:
    sn = SpinNetwork()

    # Add nodes
    n1 = sn.add_node(node_name="N1")
    n2 = sn.add_node(node_name="N2")
    n3 = sn.add_node(node_name="N3")
    n4 = sn.add_node(node_name="N4")

    # Add links with spin quantum numbers
    # Link(name=0667a0f2, nodes=(N1, N2), j=0.5)
    # Link(name=b19e3263, nodes=(N2, N3), j=1.0)
    # Link(name=911c6714, nodes=(N3, N1), j=1.5)
    # Link(name=e1c5980b, nodes=(N3, N4), j=0.5)
    l1 = sn.add_link(n1, n2, 0.5, link_name="L12")
    l2 = sn.add_link(n2, n3, 1.0, link_name="L23")
    l3 = sn.add_link(n3, n1, 1.5, link_name="L31")
    l4 = sn.add_link(n3, n4, 0.5, link_name="L34")

    print(sn)
    sn.display()

    print(f"\nLinks connected to {n3.name}:")
    for link in sn.get_links_for_node(n3):
        print(link)

    print(f"\nNeighbors of {n3.name}:")
    for neighbor in sn.get_neighbors(n3):
        print(neighbor)

    # Test adding existing node/link
    n1_again = sn.add_node(n1) # Should not create a new node
    print(f"\nTotal nodes after adding n1 again: {len(sn.nodes)}")

    l1_again = sn.add_link(n1,n2,0.5, link_name="L12_again") # Should return existing L12
    print(f"Added L12 again, received: {l1_again.name}, Original L12 name: {l1.name}")
    print(f"Total links: {len(sn.links)}")

    # Test getting node by name
    node_found = sn.get_node_by_name("N2")
    print(f"\nFound node by name 'N2': {node_found}")
    node_not_found = sn.get_node_by_name("N5")
    print(f"Found node by name 'N5': {node_not_found}")

    # Test sorting for link nodes (important for Link.__hash__ and __eq__)
    n_test_a = Node(name="A")
    n_test_b = Node(name="B")
    # Force IDs to control sort order for test
    n_test_a.id = uuid.UUID('00000000-0000-0000-0000-000000000000')
    n_test_b.id = uuid.UUID('11111111-1111-1111-1111-111111111111')

    link_ab = Link(n_test_a, n_test_b, 1.0)
    link_ba = Link(n_test_b, n_test_a, 1.0)
    print(f"\nLink AB nodes: ({link_ab.node1.name}, {link_ab.node2.name})")
    print(f"Link BA nodes: ({link_ba.node1.name}, {link_ba.node2.name})")
    print(f"Are link_ab and link_ba equal? {link_ab == link_ba}") # Should be true
    print(f"Hash of link_ab: {hash(link_ab)}, Hash of link_ba: {hash(link_ba)}") # Should be same

    s = set()
    s.add(link_ab)
    s.add(link_ba)
    print(f"Size of set after adding link_ab and link_ba: {len(s)}") # Should be 1
