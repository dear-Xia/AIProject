import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from elasticsearch import Elasticsearch

# 加载预训练的 ResNet 模型
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # 移除最后的分类层

# 图像转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取和处理图像
image = Image.open("path/to/your/image.jpg")
image = transform(image).unsqueeze(0)  # 增加 batch 维度

# 通过模型获取图像的向量表示
with torch.no_grad():
    vector = model(image).squeeze().numpy()  # 去掉 batch 维度

# 初始化 Elasticsearch 客户端
es = Elasticsearch("http://localhost:9200")

# 创建索引并存储向量
index_name = "image_vectors"
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "image_vector": {
                        "type": "dense_vector",
                        "dims": 2048  # ResNet50 输出向量维度
                    }
                }
            }
        }
    )

# 存储图像的向量表示
es.index(index=index_name, body={"image_vector": vector.tolist()})

print("Image vector stored in Elasticsearch.")
