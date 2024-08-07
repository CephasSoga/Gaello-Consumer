from datetime import datetime
from typing import List

class Node:
    """
    A class representing a node in a graph.

    Attributes:
        id (str): The unique identifier of the node.
        date (datetime): The date associated with the node.
        content (str): The content of the node.
        children (List[Node]): The child nodes of this node.
    """

    def __init__(self, id: str, content: str):
        """
        Initialize a new Node instance.

        Args:
            id (str): The unique identifier of the node.
            date (str): The date associated with the node in 'YYYY-MM-DD' format.
            content (str): The content of the node.
        """
        self.id = id
        self.timesttamp = datetime.now().isoformat()
        self.content = content

    def __str__(self) -> str:
        return f"Node(id={self.id}, date={self.timesttamp}, content={self.content})"

class Edge:
    """
    A class representing an edge in a graph.

    Attributes:
        source (Node): The source node of the edge.
        target (Node): The target node of the edge.
        relation (str): The type of relation between the nodes.
        weight (int): The weight of the edge.
    """

    def __init__(self, source: Node, target: Node, relation: str, weight: float):
        """
        Initialize a new Edge instance.

        Args:
            source (Node): The source node of the edge.
            target (Node): The target node of the edge.
            relation (str): The type of relation between the nodes.
            weight (int): The weight of the edge.
        """
        self.source = source
        self.target = target
        self.relation = relation
        self.weight = weight

    def __str__(self) -> str:
        return f"Edge(source={self.source}, target={self.target}, relation={self.relation}, weight={self.weight})"

# Example usage
if __name__ == "__main__":
    node1 = Node(id="1", content="Node 1 content")
    node2 = Node(id="2", content="Node 2 content")
    edge = Edge(source=node1, target=node2, relation="parent", weight=5)

    print(node1)
    print(node2)
    print(edge)
