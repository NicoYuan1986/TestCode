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
token = os.environ.get('TOKEN')
if uri is None:
    print("URI should not be None")
    assert False
print(fmt.format(f"Milvus uri: {uri}"))
token = "db_admin:Milvus123" if token is None else token
connections.connect(uri=uri, token=token)


collection_name = "major_compaction_collection_enable_scalar_clustering_key_1kw"
hello_milvus = Collection(name=collection_name)

num = 0
while num >= 0:
    print("Start major compaction")
    start = time.time()
    hello_milvus.compact(is_clustering=True)

    res = hello_milvus.get_compaction_state(is_clustering=True)
    print(res)

    # res = hello_milvus.get_compaction_plans()
    # print(res)

    print("waiting for compaction completed")
    hello_milvus.wait_for_compaction_completed(is_clustering=True)
    end = time.time() - start

    print("major compaction is successfully in %f s" % end)

    res = hello_milvus.get_compaction_state(is_clustering=True)
    print(res)

    res = utility.get_query_segment_info(collection_name)
    print("after major, segments number is %d"%len(res))
    num = num + 1

# res = hello_milvus.get_compaction_plans(is_clustering=True)
# print(res)
