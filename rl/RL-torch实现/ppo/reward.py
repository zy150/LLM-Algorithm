from transformers import AutoModel, AutoTokenizer
import torch
from utils import check_gpu_memory
model_path = "/models/reward/internlm/internlm2-1_8b-reward"

device = "cuda:3"
dtype = getattr(torch, "bfloat16")
reward = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,  # 允许从远程代码加载模型
    dtype=dtype
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)


chat_2 = [
    # system也会影响分数
    {"role": "system", "content": "你是一个人工智能助手。"},
    {"role": "user", "content": "Hello! What's your name?"},
    {"role": "assistant", "content": "I have no idea."}
]


print(reward)
check_gpu_memory(device)

print(reward_tokenizer(str(chat_2)))
score = reward.get_score(reward_tokenizer, chat_2)


print(score)
