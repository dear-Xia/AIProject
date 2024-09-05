from elasticsearch import Elasticsearch

# 创建 Elasticsearch 客户端
es = Elasticsearch("http://localhost:9200")

# 定义索引配置
index_settings = {
    "settings": {
        "index": {
            "number_of_shards": 5,  # 设置主分片数量
            "number_of_replicas": 1  # 设置副本数量
        }
    }
}

# 创建索引
index_name = "my_index"
es.indices.create(index=index_name, body=index_settings)

print(f"Index '{index_name}' created with {index_settings['settings']['index']['number_of_shards']} shards "
      f"and {index_settings['settings']['index']['number_of_replicas']} replicas.")
