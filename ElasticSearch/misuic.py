import librosa
import torch
import torch.nn as nn
import numpy as np
from elasticsearch import Elasticsearch

# 假设你有一个预训练的音频模型
class SimpleAudioModel(nn.Module):
    def __init__(self):
        super(SimpleAudioModel, self).__init__()
        self.fc = nn.Linear(40, 128)  # 这里是一个简单的例子

    def forward(self, x):
        return self.fc(x)

model = SimpleAudioModel()

# 加载音频并提取特征
audio, sr = librosa.load("path/to/your/audio.wav", sr=None)
mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)

# 通过模型获取音频的向量表示
with torch.no_grad():
    mfcc_tensor = torch.tensor(mfcc.T).float()
    vector = model(mfcc_tensor).mean(dim=0).numpy()  # 简单的平均池化

# 初始化 Elasticsearch 客户端
es = Elasticsearch("http://localhost:9200")

# 创建索引并存储向量
index_name = "audio_vectors"
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "audio_vector": {
                        "type": "dense_vector",
                        "dims": 128  # 音频模型输出维度
                    }
                }
            }
        }
    )

# 存储音频的向量表示
es.index(index=index_name, body={"audio_vector": vector.tolist()})

print("Audio vector stored in Elasticsearch.")
