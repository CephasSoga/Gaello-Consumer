import time
import random
import asyncio
from typing import List
from datetime import datetime, timedelta

import pika
from pika.exceptions import AMQPError, AMQPChannelError, AMQPConnectionError

from graph._types import Node
from graph.singleshot import insert_and_update
from utils_consumer.logger import Logger

RETRY_COUNT = 5
DELAY_SEC = 5

class Consumer:
    def __init__(self, queue_name: str, host: str = 'localhost', port: int = 5672, prefetch_count: int = 10000, batch_size: int = 10000, batch_interval: int = 10, **kwargs):
        """
        Initializes a new instance of the Consumer class.

        Args:
            queue_name (str): The name of the queue to consume messages from.
            host (str, optional): The hostname of the RabbitMQ server. Defaults to 'localhost'.
            port (int, optional): The port number of the RabbitMQ server. Defaults to 5672.
            prefetch_count (int, optional): The number of messages to prefetch. Defaults to 10000.
            batch_size (int, optional): The size of the batch to process. Defaults to 10000.
            batch_interval (int, optional): The interval in seconds between batch processing. Defaults to 10.
            **kwargs: Additional keyword arguments to pass to the ConnectionParameters constructor.

        Returns:
            None

        Initializes the following instance variables:
            - queue_name (str): The name of the queue to consume messages from.
            - host (str): The hostname of the RabbitMQ server.
            - port (int): The port number of the RabbitMQ server.
            - connection (None): The RabbitMQ connection.
            - channel (None): The RabbitMQ channel.
            - prefetch_count (int): The number of messages to prefetch.
            - batch_size (int): The size of the batch to process.
            - batch_interval (timedelta): The interval between batch processing.
            - last_batch_time (datetime): The timestamp of the last batch processing.
            - logger (Logger): The logger instance for the Graph Consumer.
            - connection_params (ConnectionParameters): The connection parameters for the RabbitMQ server.
            - nodes (List[Node]): The list of nodes to process.
            - buffer (List[Node]): The buffer for storing nodes before processing.
            - loop (asyncio.AbstractEventLoop): The event loop for asynchronous operations.
        """
        self.queue_name = queue_name
        self.host = host
        self.port = port
        self.connection = None
        self.channel = None
        self.prefetch_count = prefetch_count
        self.batch_size = batch_size
        self.batch_interval = timedelta(seconds=batch_interval)
        self.last_batch_time = datetime.now()

        self.logger = Logger('Graph Consumer')
        
        # Configure connection parameters
        self.connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            connection_attempts=10,
            retry_delay=5,
            **kwargs
        )

        self.nodes: List[Node] = []
        self.buffer: List[Node] = []

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def connect(self, retries: int = RETRY_COUNT, delay: int = DELAY_SEC):
        """
        Connects to a RabbitMQ server and establishes a connection and channel.

        Args:
            retries (int, optional): The number of times to retry connecting to the RabbitMQ server. Defaults to RETRY_COUNT.
            delay (int, optional): The number of seconds to wait between retries. Defaults to DELAY_SEC.

        Raises:
            ConnectionError: If unable to connect to the RabbitMQ server after multiple attempts.

        Returns:
            None
        """
        for attempt in range(retries):
            try:
                self.connection = pika.BlockingConnection(self.connection_params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue_name)
                self.channel.basic_qos(prefetch_count=self.prefetch_count)  # Set prefetch count
                self.logger.log("info", f"Amqp > Connection:: Connected to RabbitMQ on {self.host}:{self.port}")
                return
            except Exception as e:
                self.logger.log("error", f"Amqp > Connection:: Error connecting to RabbitMQ (attempt {attempt + 1}/{retries})", e)
                time.sleep(delay)
        raise ConnectionError("Unable to connect to RabbitMQ after multiple attempts.")


    def callback(self, ch, method, properties, body):
        """
        Callback function to handle incoming messages from RabbitMQ.
        
        Args:
            ch (pika.channel.Channel): The RabbitMQ channel.
            method (pika.spec.Basic.Deliver): The delivery method.
            properties (pika.spec.BasicProperties): The properties of the message.
            body (bytes): The message body.
        
        Raises:
            Exception: If there is an error processing the message.
        
        Returns:
            None
        """
        try:
            id = f"{random.randint(0, int(1e3))}" + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
            date = datetime.now().date().isoformat()
            timestamp = datetime.now().time().strftime("%H:%M:%S")
            content = body.decode()
            node = Node(id, date, timestamp, content)
            self.buffer.append(node)
            self.logger.log("info", f"Amqp > Buf.:: Received message and added to buffer. Buffer length: {len(self.buffer)}")

            # Check if we should process the batch
            if len(self.buffer) >= self.batch_size and datetime.now() - self.last_batch_time >= self.batch_interval:
                self.process_batch()
                self.last_batch_time = datetime.now()

        except Exception as e:
            self.logger.log("error", f"Amqp > Buf.:: Error processing message.", e)
            # Decide whether to requeue the message or handle the error in another way

    def process_batch(self):
        """
        Process the batch of messages.

        This function processes a batch of messages by extending the `self.nodes` list with the messages in `self.buffer`. 
        It logs an informational message with the number of messages being processed. 
        It then calls the `insert_and_update` function to insert and update the processed messages in the database. After processing, the buffer is cleared.

        Parameters:
            None

        Returns:
            None

        Raises:
            Exception: If there is an error processing the batch.

        """
        try:
            # Process the batch of messages
            if self.buffer:
                self.nodes.extend(self.buffer)
                self.logger.log("info", f"Amqp > Proc.:: Processing batch of {len(self.buffer)} messages")
                insert_and_update(self.buffer)
                # Clear the buffer after processing
                print("Buffer length: {}".format(len(self.buffer)))
                self.buffer.clear()
        except Exception as e:
            self.logger.log("error", "Amqp > Proc.:: Error processing batch.", e)

    def consume(self):
        """
        Consume messages from a RabbitMQ queue.

        This method connects to the RabbitMQ server, consumes messages from the specified queue, and processes them using the `callback` method. 
        It sets the `auto_ack` parameter to `True` to automatically acknowledge the received messages. 
        The method continues consuming messages until it is interrupted by the user or an error occurs.

        Parameters:
            None

        Raises:
            KeyboardInterrupt: If the consumer is interrupted by the user.
            AMQPError, AMQPChannelError, AMQPConnectionError: If there is an error consuming messages.

        Returns:
            None
        """
        try:
            self.connect()
            self.logger.log("info", f"Amqp > Consumption:: Consuming messages from queue '{self.queue_name}'...")
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.callback, auto_ack=True)
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.log("info", "Amqp > Consumption:: Consumer interrupted by user")
        except (AMQPError, AMQPChannelError, AMQPConnectionError) as amqp_e:
            self.logger.log("error", f"Amqp > Consumption:: Error consuming messages.", amqp_e)
        finally:
            self.close()

    def close(self):
        """
        Close the connection to the RabbitMQ server.

        This function attempts to close the connection to the RabbitMQ server. If the connection is open,
        it will be closed. If an error occurs during the closing process, an error message will be logged.

        Parameters:
            None

        Returns:
            None

        Raises:
            AMQPError: If there is an error closing the connection.

        """
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
                self.logger.log("info", "Amqp > Connection:: Connection to RabbitMQ closed")
        except (AMQPError, AMQPChannelError, AMQPConnectionError) as amqp_e:
            self.logger.log("error", f"Amqp > Connection:: Error closing connection", amqp_e)
