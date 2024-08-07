from consumer.ops import Consumer
from utils_consumer.envHandler import getenv

PREFETCH_COUNT = 5
BATCH_SIZE = 5
BATCH_INTERVAL = 0

queue_name = getenv("RMQ_QUEUE", "gaello_queue")
print(f"Queue name: {queue_name}")
host = getenv("RMQ_HOST", "amqp-connection")
print(f"Host: {host}")

if __name__ == "__main__":
    
    consumer = Consumer(
        queue_name=queue_name, 
        host=host,  
        prefetch_count=PREFETCH_COUNT, 
        batch_size=BATCH_SIZE, 
        batch_interval=BATCH_INTERVAL)
    
    loop = consumer.loop
    
    try:
        consumer.consume()
    finally:
        consumer.loop.close()