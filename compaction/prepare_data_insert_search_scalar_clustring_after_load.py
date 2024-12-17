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


default_fields = [
    FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="key", dtype=DataType.INT64, is_clustering_key=True),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="var", dtype=DataType.VARCHAR, max_length=10000, is_primary=False),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
default_schema = CollectionSchema(fields=default_fields, description="test clustering-key collection")
collection_name = "major_compaction_collection_enable_scalar_clustering_key_1kw"

if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.drop()
    print("drop the original collection")

hello_milvus = Collection(name=collection_name, schema=default_schema)
print("success to create collection %s " % collection_name)


nb = 10000

rng = np.random.default_rng(seed=19530)
random_data = rng.random(nb).tolist()

_len = int(20)
_str = string.ascii_letters + string.digits
_s = _str
print("_str size ", len(_str))

for i in range(int(_len / len(_str))):
    _s += _str
    print("append str ", i)
values = [''.join(random.sample(_s, _len - 1)) for _ in range(nb)]
index = 0
while index < 1000:
    # insert data
    vec_data = [[random.random() for _ in range(dim)] for _ in range(nb)]
    data = [
        [index * nb + i for i in range(nb)],
        [random.randint(0, 1004) for i in range(nb)],
        random_data,
        values,
        vec_data,
    ]
    start = time.time()
    res = hello_milvus.insert(data)
    end = time.time() - start
    print("insert %d %d done in %f" % (index, nb, end))
    index += 1
    # hello_milvus.flush()

print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

# 4. create index
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 100},
}

hello_milvus.create_index("embeddings", index)


hello_milvus.load()

print("Start compaction")
start = time.time()
hello_milvus.compact()

res = hello_milvus.get_compaction_state()
print(f"compaction state: {res}")

res = hello_milvus.get_compaction_plans()
print(f"compaction state: {res}")

print("waiting for compaction completed")
hello_milvus.wait_for_compaction_completed()
end = time.time() - start

print("compaction is successfully in %f s" % end)

nb = 1
vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
nq = 1
default_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
res1 = hello_milvus.search(vectors[:nq], "embeddings", default_search_params, 10, "count >= 0")

print("search ... ")
print(res1[0].ids)
res1 = hello_milvus.query("count >= 0", output_fields=["count(*)"])
print(res1)


print("collection num is %d" % hello_milvus.num_entities)
res = utility.get_query_segment_info(collection_name)
logger.info("before major, segments number is %d" % len(res))
# logger.info(res)
