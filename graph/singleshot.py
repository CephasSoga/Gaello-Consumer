from typing import List

from graph.graph import Graph
from graph._types import Node
from utils_consumer.envHandler import getenv

def insert_and_update(
        nodes: List[Node],
        limit: int = 5000,
        uri = getenv("NEO4J_URI"),
        username = "neo4j",
        password = getenv("NEO4J_PWD"),
    ):
    """
    Insert a list of nodes into the graph database and update the graph with the new nodes.

    Args:
        nodes (List[Node]): The list of nodes to be inserted into the graph.
        limit (int, optional): The maximum number of nodes to be inserted in a single batch. Defaults to 5000.
        uri (str, optional): The URI of the graph database. Defaults to env var NEO4J_URI.
        username (str, optional): The username for authentication. Defaults to "neo4j".
        password (str, optional): The password for authentication. Defaults to env var NEO4J_PWD.

    Returns:
        None
    """

    graph = Graph(uri, username, password)

    graph._batch_insert(nodes)
    graph.update(limit)