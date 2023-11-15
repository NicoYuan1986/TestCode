from pymilvus import CollectionSchema, FieldSchema, Collection, connections, DataType, Partition, utility
import random
import numpy as np
import pandas as pd
import logging as log
import time
import sys

args = sys.argv
HOST = args[1]
PORT = args[2]
connections.connect(host=HOST, port=PORT)
log.basicConfig(filename='test_binlog.log', level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

collection_name = "test_binlog_true"
dim = 768
limit = 5
nq = 2
ef = 64


def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields)
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    log.info(f"[Record]Start to create collection - {collection_name}")
    c = Collection(collection_name, schema=schema)
    log.info(f"[Record]collection schema : {c.schema}")
    return c


def create_index_load(collection: Collection):
    if collection.has_index():
        collection.release()
        collection.drop_index()
    # use default index: {"M": 18,"efConstruction": 240,"index_type": "HNSW", "metric_type": "IP"}
    index_params = {"M": 18, "efConstruction": 360, "index_type": "HNSW", "metric_type": "COSINE"}
    collection.create_index("float_vector", index_params)
    log.info(f"[Record]Succeed to create index: {collection.index().params}")
    collection.load()
    log.info(f"[Record]Succeed to load collection: {collection_name}")


def insert_data(collection: Collection, start=0, per_nb=10000):
    vectors = [[random.random() for _ in range(dim)] for _ in range(per_nb)]
    data = pd.DataFrame({'id': [_id for _id in range(start, start + per_nb)],
                         'float_vector': vectors})
    t0 = time.time()
    collection.insert(data)
    timestamp = time.time() - t0
    cost = "{:.2f}".format(timestamp)
    count = collection.num_entities
    log.info(f"insert {per_nb} entities in {cost}s, all entities: {count}")


def collection_search(collection: Collection):
    search_params = {"metric_type": "COSINE", "params": {"ef": ef}}
    vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
    t0 = time.time()
    collection.search(vectors, "float_vector", search_params, limit)
    latency = time.time() - t0
    latency = float("{:.2f}".format(latency * 1000))
    log.info(f"search in {latency} ms")
    return latency


def insert_search_loop():
    collection = create_collection()
    create_index_load(collection)
    start = 0
    for i in range(1, 10):
        # insert
        t0 = time.time()
        log.info(f"\n\n round {i}, start to insert data\n")
        while time.time() - t0 < 900:
            insert_data(collection, start)
            start += 10000
        # search
        log.info(f"\n\n round {i}, start to search - 1\n")
        t0 = time.time()
        latencies = []
        search_times = 0
        while time.time() - t0 < 300:
            latency = collection_search(collection)
            latencies.append(latency)
            search_times += 1
        latency = sum(latencies) / len(latencies)
        latency = "{:.2f}".format(latency)
        log.info(f"Search finished, total {search_times} times, latency {latency} ms")
        # sleep
        log.info(f"\n\n round {i}, start to sleep\n")
        time.sleep(600)
        # search
        log.info(f"\n\n round {i}, start to search - 2\n")
        t0 = time.time()
        latencies = []
        search_times = 0
        while time.time() - t0 < 300:
            latency = collection_search(collection)
            latencies.append(latency)
            search_times += 1
        latency = sum(latencies) / len(latencies)
        latency = "{:.2f}".format(latency)
        log.info(f"Search finished, total {search_times} times, latency {latency} ms")


if __name__ == "__main__":
    insert_search_loop()
