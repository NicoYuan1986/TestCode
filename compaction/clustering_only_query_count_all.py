import os
import time
import random
import string
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
dim = 128


print(fmt.format("start connecting to Milvus"))
uri = os.environ.get('MILVUS_URI')
if uri is None:
    print("URI should not be None")
    assert False
print(fmt.format(f"Milvus uri: {uri}"))
connections.connect(uri=uri, token="db_admin:Milvus123")


collection_name = "major_compaction_collection_enable_scalar_clustering_key_1kw"


hello_milvus = Collection(name=collection_name)

print("Start compaction")
start = time.time()
hello_milvus.compact()

res = hello_milvus.get_compaction_state()
print(res)

res = hello_milvus.get_compaction_plans()
print(res)

print("waiting for compaction completed")
hello_milvus.wait_for_compaction_completed()
end = time.time() - start

res = hello_milvus.get_compaction_state()
print(res)

print("compaction is successfully in %f s" % end)


start = time.time()
duration = 0

res = utility.get_query_segment_info(collection_name)
print("before major, segments number is %d" % len(res))
dim = 128

nb = 1
vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]

while duration >= 0:
    duration = time.time() - start
    default_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    limit = 10
    res1 = hello_milvus.search(vectors[:1], "embeddings", default_search_params, limit,
                               "key == 100")
    print(len(res1))
    res = hello_milvus.query("count>=0", output_fields=["count(*)"])
    print(res[0]['count(*)'])
    assert res[0]['count(*)'] == 10000000
    duration = duration + 1
    time.sleep(1)
