import faiss
import numpy as np
from elasticsearch import Elasticsearch

# 初始化 Elasticsearch 客户端
es = Elasticsearch("http://localhost:9200")

# 假设我们有一组向量
vectors = np.random.random((1000, 128)).astype('float32')  # 1000 个 128 维向量

# 定义 PQ 参数
d = vectors.shape[1]  # 向量维度
m = 8  # 子向量数量
k = 256  # 每个子量化器的码字数量

# 创建 PQ 量化器
quantizer = faiss.IndexFlatL2(d)  # 训练 PQ 之前需要一个量化器
pq = faiss.IndexPQ(d, m, k)
pq.train(vectors)

# 压缩向量
pq.add(vectors)
compressed_vectors = pq.codes  # 获取压缩后的向量

# 你可以将这些压缩后的向量作为二进制数据存储到 Elasticsearch 中

# 假设我们要创建一个索引来存储这些向量
index_name = "compressed_vectors_index"
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

es.indices.create(
    index=index_name,
    body={
        "mappings": {
            "properties": {
                "vector": {
                    "type": "binary"
                }
            }
        }
    }
)

# 将压缩后的向量存储到 Elasticsearch 中
for i, vec in enumerate(compressed_vectors):
    es.index(
        index=index_name,
        id=i,
        body={
            "vector": vec.tobytes()  # 将 numpy 数组转换为字节串存储
        }
    )

print(f"Stored {len(compressed_vectors)} compressed vectors in Elasticsearch.")



# 查询压缩后的向量
# 查询示例：计算查询向量与存储向量的相似性
query_vector = np.random.random((1, 128)).astype('float32')  # 一个随机的查询向量
pq.add(query_vector)
compressed_query_vector = pq.codes[0]  # 压缩查询向量

# 从 Elasticsearch 检索向量（假设这里只获取前 10 个结果）
results = es.search(index=index_name, size=10, body={"query": {"match_all": {}}})

# 计算查询向量与返回向量的距离
distances = []
for hit in results['hits']['hits']:
    stored_vector = np.frombuffer(hit['_source']['vector'], dtype=np.uint8)  # 从字节串转换回 numpy 数组
    distance = faiss.pq_reconstruct_distance(pq, stored_vector, compressed_query_vector)
    distances.append((hit['_id'], distance))

# 按距离排序
distances.sort(key=lambda x: x[1])

# 输出最相似的向量
print("Top similar vectors:")
for vector_id, distance in distances:
    print(f"Vector ID: {vector_id}, Distance: {distance}")

