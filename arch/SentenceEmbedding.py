from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
import time
import re
import torch
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 加载节点描述
df = pd.read_csv("GBA/station_descriptions.csv")  # 包含 'description' 列

# 2. 加载模型（推荐先用小模型测试）
model = SentenceTransformer('offline_models/multi-qa-MiniLM-L6-cos-v1')  # 也可以换成 'all-mpnet-base-v2'

# 3. 生成嵌入
start_time = time.time()
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)
end_time = time.time()
total_time = end_time - start_time
print(embeddings.shape)
print(f"Total embedding time: {total_time:.4f} seconds")
# 4. 保存嵌入
np.save("/GBA/node_embeddings_multi-qa.npy", embeddings) # (N, 384)

