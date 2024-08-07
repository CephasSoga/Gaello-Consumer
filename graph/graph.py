from typing import List, Dict, Any

from tqdm import tqdm
from neo4j import Transaction

from graph.cypher import Cypher
from graph._types import Node, Edge
from graph.handler import Neo4jHandler
from graph.embeddings import OptimizedHashEmbedding
from utils_consumer.logger import Logger

class Graph:
    """
    A class to represent and interact with a graph in Neo4j.

    Attributes:
        handler (Neo4jHandler): The handler for Neo4j database operations.
        connected (List[tuple[Node]]): List of connected node pairs.
    """
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the Graph with a Neo4j handler.

        Args:
            uri (str): The URI of the Neo4j database.
            username (str): The username for authentication.
            password (str): The password for authentication.

        Initializes the following instance variables:
            - loggger (Logger): A logger instance for logging messages.
            - handler (Neo4jHandler): A handler for Neo4j database operations.
            - hash_embedding (OptimizedHashEmbedding): An instance of the OptimizedHashEmbedding class.
            - embeddings_dict (Dict[str, Any]): A dictionary to store embeddings.
        """

        self.loggger = Logger('Graph Generator')

        self.handler = Neo4jHandler(uri, username, password)
        self.hash_embedding = OptimizedHashEmbedding(10000, 300)
        self.embeddings_dict = {}

    async def create_index(self):
        """
        Asynchronously creates an index on the Node label.

        This function logs an informational message indicating that an index is being created on Node labels.
        It then creates a Cypher query to create an index on the Node label using the 'id', 'date', and 'timestamp'
        properties. The query is executed using the 'async_run_write_query' method of the 'handler' instance.
        The result of the query is logged as an informational message.

        If an exception occurs during the creation of the index, an error message is logged along with the exception.
        The function returns None.

        Returns:
            None: If an exception occurs during the creation of the index.
        """

        self.loggger.log("info", "Graph > Write:: Creating index on Node labels...")

        try:
            create_index_query = Cypher("""
                CREATE INDEX FOR (n:Node) ON (n.id, n.date, n.timestamp);
            """)

            result =  await self.handler.async_run_write_query(create_index_query)

            self.loggger.log("info", f"Graph > Write:: Index created on Node labels: {result}")

        except Exception as e:
            self.loggger.log("error", "Error creating index on Node labels", e)
            return None


    async def create_constraint(self):
        """
        Asynchronously creates a uniqueness constraint on node IDs.

        This function logs an informational message indicating that a uniqueness constraint is being created on Node IDs.
        It then creates a Cypher query to create a uniqueness constraint on the Node ID property. The query is executed
        using the 'async_run_write_query' method of the 'handler' instance. The result of the query is logged as an
        informational message.

        If an exception occurs during the creation of the constraint, an error message is logged along with the exception.
        The function returns None.

        Returns:
            None: If an exception occurs during the creation of the constraint.
        """

        self.loggger.log("info", "Graph > Write:: Creating uniqueness constraint on Node IDs...")

        try:
            create_constraint_query = Cypher("""
            CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE
            """)

            result = await self.handler.async_run_write_query(create_constraint_query)

            self.loggger.log("info", "Graph > Write:: Uniqueness constraint created on Node IDs: {result}")

        except Exception as e:
            self.loggger.log("error", "Graph > Write:: Error creating uniqueness constraint on Node IDs", e)
            return None

    async def insert(self, node: Node):
        """
        Inserts a new node into the database.

        Args:
            node (Node): The node to be inserted.

        Returns:
            Any: The result of the insert query.
        """
        query = Cypher("""
            CREATE (n:Node {id: $id, date: $date, timestamp: $timestamp, content: $content})
            RETURN n
            """)
        
        params = {
            "id": node.id,
            "date": node.date,
            "timestamp": node.timestamp,
            "content": node.content
        }
        return await self.handler.async_run_write_query(query, **params)
    
    def _batch_insert(self, nodes: List[Node]):
        """
        Batch insert nodes into the database.

        Args:
            nodes (List[Node]): The nodes to insert.

        Returns:
            None: If an error occurs during the insertion process.

        This method logs an informational message indicating that nodes are being inserted into the database.
        It then creates a Cypher query to insert the nodes into the database using the 'id', 'date', 'timestamp',
        and 'content' properties. The query is executed using the 'run_write_query' method of the 'handler' instance.
        The result of the query is logged as an informational message.

        If an exception occurs during the insertion process, an error message is logged along with the exception.
        The method returns None.
        """

        self.loggger.log("info", "Graph > Write:: Inserting nodes...")

        try:
            query = Cypher("""
                UNWIND $nodes AS node
                CREATE (n:Node {id: node.id, date: node.date, timestamp: node.timestamp, content: node.content})
                """)
            params = {
                "nodes": [{"id": node.id, "date": node.date, "timestamp": node.timestamp, "content": node.content} for node in nodes]
            }
            result = self.handler.run_write_query(query, **params)

            self.loggger.log("info", "Graph > Write:: Done inserting nodes.")

        except Exception as e:
            self.loggger.log("error", "Graph > Write:: Error inserting Nodes", e)
            return None

    @staticmethod
    def _perform_within_tx(tx: Transaction, query: Cypher, **params):
        """
        Perform a query within a transaction.
        Ideal for batch operations.

        Args:
            tx (Transaction): The transaction object.
            node1_id (str): The ID of the first node.
            node2_id (str): The ID of the second node.
            relation (str): The relation type.
        
        """
        tx.run(query.exec_cmd, **params)

    def precomputed_embeddings(self, nodes: List[Node]) -> Dict[str, Any]:
        """
        Compute precomputed embeddings for nodes.

        Args:
            nodes (List[Node]): The list of nodes.

        Returns:
            Dict[str, Any]: A dictionary mapping node IDs to their embeddings. If an error occurs during the embedding process, an empty dictionary is returned.
        """

        embeddings_dict = {}
        try:
            for node in tqdm(nodes, desc="Embedding..."):
                embedding = self.hash_embedding.compute_embeddings_from_str(node.content)
                embeddings_dict[node.id] = embedding
            return embeddings_dict
        except Exception as e:
            self.loggger.log("error", "Graph > Embedding:: Error occurred while embedding", e)
            return {}

    def _match_nodes_batch(self, nodes: List[Node], embeddings_dict: Dict):
        """
        Connect nodes based on a query result.

        Args:
            result (List[Dict[str, Any]]): The query result containing nodes.

        Returns:
            List[Edge]: The list of created edges.
        """
        relation_type = "SIMILAR_TO"
        try:
            total_iter = len(nodes)
            indices: set[int] = set()
            pointed_at: set[Node] = set()
            exhausted = False

            with tqdm(total=total_iter, desc="Matching Nodes...") as pbar:
                i = 0
                while i < total_iter and not exhausted:

                    first = nodes[i]
                    indices.add(i)
                    remaining_nodes = [node for idx, node in enumerate(nodes) if idx not in indices]

                    if not remaining_nodes:
                        exhausted = True
                        break

                    for second in remaining_nodes:
                        if second in pointed_at:
                            continue
                        is_match = self.hash_embedding.compute_similarity_from_embeddings(embeddings_dict[first.id], embeddings_dict[second.id])
                        if is_match > 0.4:
                            relation_weight = is_match
                            edge = Edge(first, second, relation=relation_type, weight=relation_weight)
                            pointed_at.add(second)
                            yield edge

                    indices.add(i)
                    pbar.update(1)
                    i += 1

        except Exception as e:
            print(f"Error occurred while matching: {e}")
            raise
    
    def _implement_edges(self, edges: List[Edge]):
        """
        Implement edges in the graph.

        Args:
            edges (List[Edge]): The list of edges to implement.

        Returns:
            Any: The result of the edge implementation query.
        """
        query = Cypher("""
        UNWIND $edges AS edge
        MATCH (a) WHERE a.id = edge.start
        MATCH (b) WHERE b.id = edge.end
        MERGE (a)-[r:RELATION]->(b)
        SET r.type = edge.type, r.weight = edge.weight
        """)

        params = {
            "edges": [{"start": edge.source.id, "end": edge.target.id, "type": edge.relation, "weight": edge.weight} for edge in edges]
        }
        return self.handler.run_static(
            "write_transaction",
            self._perform_within_tx, query=query, **params
        )

    def connect(self, result: List[Dict[str, Any]], batch_size: int=100) -> List[Edge]:
        """
        Connect nodes based on a query result.

        Args:
            result (List[Dict[str, Any]]): The query result containing nodes.

        Returns:
            List[Edge]: The list of created edges.
        """
        nodes = [
            Node(
                record["n"]["id"],
                record["n"]["date"],
                record["n"]["timestamp"],
                record["n"]["content"]
            ) for record in result
        ]

        embeddings  = self.precomputed_embeddings(nodes)

        if not embeddings:
            return
        
        edges = self._match_nodes_batch(nodes, embeddings)

        edges = [edge for edge in edges if edge]

        batches = [edges[i:i + batch_size] for i in range(0, len(edges), batch_size)]
        
        _ = [self._implement_edges(batch) for batch in batches]

        return 0

    async def gather_top_nodes(self, offset, limit: int=100):
        """
        Gathers the top nodes from the Neo4j database based on the date amd tomestamp fields.

        Args:
            offset (int): The number of nodes to skip before starting to return nodes.
            limit (int, optional): The maximum number of nodes to return. Defaults to 100.

        Returns:
            List[Dict[str, Any]] or None: A list of dictionaries representing the nodes, where each dictionary contains the node properties.
            If an error occurs, returns None.

        Raises:
            Exception: If any error occurs during the database query.
        """
        try:
            query = Cypher(
            """
            MATCH (n:Node)
            RETURN n
            ORDER BY n.date, n.timestamp DESC
            SKIP $offset
            LIMIT $limit
            """
            )

            result = await self.handler.async_run_read_query(query, offset=offset, limit=limit)
            return result
        
        except Exception as e:
            print(e)
            return None
    
    async def get_connected_nodes(self, node_id:str, depth:int, timestamp_threshold=None):
        """
        Asynchronously retrieves all nodes connected to a given node within a specified depth.

        Args:
            node_id (str): The ID of the root node.
            depth (int): The maximum depth to traverse from the root node.
            timestamp_threshold (datetime.datetime, optional): A timestamp threshold to filter nodes. Defaults to None.

        Returns:
            List[Dict[str, Any]] or None: A list of dictionaries representing the connected nodes, where each dictionary contains the node properties.
            If an error occurs, returns None.
        """

        try:
            query = Cypher(
            f"""
            MATCH (root:Node {{id: {node_id}}})-[*..{depth}]-(connected:Node)
            WHERE root <> connected
            RETURN DISTINCT connected
        
            """
            )
            return await self.handler.async_run_read_query(query)
        
        except Exception as e:
            return None

    def update(self, limit=1500):
        """
        Update the graph by connecting nodes.

        Args:
            limit (int): The number of nodes to process.

        Returns:
            tuple[str, List[Edge]]: The status and list of created edges.
        """
        self.loggger.log('info', "Graph > Update:: Updating graph...")
        try:
            query = Cypher("""
                MATCH (n:Node)
                RETURN n
                ORDER BY n.date, n.timestamp DESC
                LIMIT $limit
            """)
            _ = self.handler.run_read_query(query, callback=self.connect, limit=limit)

            self.loggger.log("info", "Graph > Update:: Done updating graph.")
            return "UPDATE_SUCCESS"
        except Exception as e:
            self.loggger.log("error", "Graph > Update:: Error updating graph", e)
            return "UPDATE_FAILURE"
        finally:
            self.handler.close()
