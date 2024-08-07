import asyncio
from typing import Callable, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from neo4j import GraphDatabase

from graph.cypher import Cypher

class Neo4jHandler:
    """
    A handler class for interacting with a Neo4j database both synchronously and asynchronously.

    Attributes:
        driver (neo4j.Driver): The Neo4j driver for database connections.
        executor (ThreadPoolExecutor): Executor for running tasks in a separate thread.
    """

    def __init__(self, uri, username, password):
        """
        Initialize the Neo4jHandler with the given connection details.

        Args:
            uri (str): The URI of the Neo4j database.
            username (str): The username for authentication.
            password (str): The password for authentication.
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.executor = ThreadPoolExecutor(max_workers=5)

    def close(self):
        """Close the database connection and shut down the executor."""
        self.driver.close()
        self.executor.shutdown()

    def run_simple_query(self, cypher_query: Cypher, **params):
        """
        Run a simple query.

        Args:
            cypher_query (Cypher): The Cypher query to execute.
            **params: Parameters for the Cypher query.

        Returns:
            Any: The result of the query.
        """
        with self.driver.session() as session:
            return session.run(cypher_query.query, **params)

    def run_static(self, mode: str, func: Callable, *args, **kwargs):
        """
        Run a static transaction.

        Args:
            mode (str): The mode of transaction ('write_transaction' or 'read_transaction').
            func (Callable): The function to execute within the transaction.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the transaction.
        """
        with self.driver.session() as session:
            if mode == 'write_transaction':
                return session.write_transaction(func, *args, **kwargs)
            elif mode == 'read_transaction':
                return session.read_transaction(func, *args, **kwargs)
            else:
                raise ValueError(f"Invalid mode '{mode}'")

    def run_write_query(self, cypher_query: Cypher, **params):
        """
        Run a write query.

        Args:
            cypher_query (Cypher): The Cypher query to execute.
            **params: Parameters for the Cypher query.

        Returns:
            Any: The result of the query.
        """
        with self.driver.session() as session:
            return session.write_transaction(cypher_query.execute, **params)

    def run_read_query(self, cypher_query: Cypher, callback: Callable[[Any], Any] = None, **params):
        """
        Run a read query.

        Args:
            cypher_query (Cypher): The Cypher query to execute.
            callback (Optional[Callable[[Any], Any]]): A callback function to process the result.
            **params: Parameters for the Cypher query.

        Returns:
            Any: The result of the query, optionally processed by the callback.
        """
        with self.driver.session() as session:
            result = session.read_transaction(lambda tx: tx.run(cypher_query.exec_cmd, **params).data())
            if callback is not None:
                return callback(result)
            else:
                return result

    async def async_run_static(self, mode: str, func: Callable, *args, **kwargs):
        """
        Asynchronously run a static transaction.

        Args:
            mode (str): The mode of transaction ('write_transaction' or 'read_transaction').
            func (Callable): The function to execute within the transaction.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the transaction.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, partial(self.run_static, mode, func, *args, **kwargs))

    async def async_run_write_query(self, cypher_query: Cypher, **params):
        """
        Asynchronously run a write query.

        Args:
            cypher_query (Cypher): The Cypher query to execute.
            **params: Parameters for the Cypher query.

        Returns:
            Any: The result of the query.
        """
        loop = asyncio.get_running_loop()
        func = partial(self.run_write_query, cypher_query, **params)
        return await loop.run_in_executor(self.executor, func)

    async def async_run_read_query(self, cypher_query: Cypher, callback: Callable[[Any], Any] = None, **params):
        """
        Asynchronously run a read query.

        Args:
            cypher_query (Cypher): The Cypher query to execute.
            callback (Optional[Callable[[Any], Any]]): A callback function to process the result.
            **params: Parameters for the Cypher query.

        Returns:
            Any: The result of the query, optionally processed by the callback.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, partial(self.run_read_query, cypher_query, None, **params))
        if callback and asyncio.iscoroutinefunction(callback):
            return await callback(result)
        elif callback and not asyncio.iscoroutinefunction(callback):
            return callback(result)
        else:
            return result
