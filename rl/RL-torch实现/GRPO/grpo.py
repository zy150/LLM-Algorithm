from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from torch import nn
import math
import swanlab
from data import GSM8KRatingUtils

@dataclass
class BaseConfig:
  device: str = 'cuda:2'
  dtype: str = 'bfloat16'
  actor_model_path: str = 'models/Qwen/Qwen3-0.6B'
  refer_device: str = 'cuda:0'
  critic_model_path : str = 'models/Qwen/Qwen3-0.6B'

@dataclass
class InferConfig(BaseConfig):
    max_new_tokens: int = 400
    temperature: float = 1.1
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    top_k: int = 50
    

@dataclass
class GRPOConfig:
  group_size: int = 4    # 一组采几个样
  learning_rate: float = 5e-6 # 建议稍微小一点
  batch_size: int = 2      # 这是 Prompt 的 Batch Size
  beta: float = 0.04       # KL 惩罚系数
  clip_ratio: float = 0.2  # clip 范围
  epoch:int = 2
@dataclass
class MainConfig:
    # 将不同模块的配置嵌套在一起
    base: BaseConfig = field(default_factory=BaseConfig)
    infer: InferConfig = field(default_factory=InferConfig,metadata={'help': 'infer config'})
    grpo: GRPOConfig = field(default_factory=GRPOConfig,metadata={'help': 'grpo config'})
    seed: int = 42
    
  
  
class PromptDataset(Dataset):
  def __init__(self, data_path, tokenizer):
    SYSTEM_PROMPT = """
    按照如下格式回答问题：
    <think>
    你的思考过程
    </think>
    <answer>
    你的回答
    </answer>
    """
    data = load_dataset(data_path,"main",split='train')
    self.tokenizer = tokenizer
    self.prompt_input = []
    self.answers = []
    data = data.filter(lambda x: len(x["question"]) < 200)
    for p in data:
      self.prompt_input.append(self.tokenizer.apply_chat_template(
              [
                  {"role": "system", "content": f"你是一个人工智能助手。{SYSTEM_PROMPT}"},
                  {"role": "user",   "content": p["question"]}
              ],
              add_generation_prompt=True,
              tokenize=False,# 输出为文本，True为token
          ))
      self.answers.append(p['answer'])
  def __getitem__(self, ix):
    return {'question': self.prompt_input[ix], 'answer': self.answers[ix]}
  
  def __len__(self):
    return len(self.answers)
  
    
    
def init_grpo_models(cfg: MainConfig):
  device = cfg.base.device
  actor = AutoModelForCausalLM.from_pretrained(
    cfg.base.actor_model_path, dtype=getattr(torch, cfg.base.dtype), trust_remote_code=True
  ).to(device)
  
  ref = AutoModelForCausalLM.from_pretrained(
    cfg.base.actor_model_path, dtype=getattr(torch, cfg.base.dtype), trust_remote_code=True
  ).to(cfg.base.refer_device)
  
  return actor, ref
    
class GRPO():
  def __init__(self,log,data_path="openai/gsm8k"):
    
    self.cfg = MainConfig()
    
    self.actor, self.ref = init_grpo_models(self.cfg)
    
    self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base.actor_model_path,padding_side='left')
    
    self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.grpo.learning_rate)
    self.actor.train()
    self.ref.eval()
    
    self.log = log
    self.actor_device = self.cfg.base.device
    self.ref_device = self.cfg.base.refer_device


    for param in self.ref.parameters():
      param.requires_grad = False # 冻结参数
      
      
    self.dataloader = DataLoader(PromptDataset(data_path, self.tokenizer), batch_size=self.cfg.grpo.batch_size)
    if self.log:
      swanlab.login(api_key="73s0aS3od1jIatxssfJJ8", save=True)
      # https://docs.swanlab.cn/api/py-init.html
      swanlab.init(
            project="LLM-RL",
            config=self.__dict__,
            experiment_name="grpo-Qwen3-0.6B",
            description="grpo练习，纯torch实现"
        )
    
  def train(self):
    pbar = tqdm(self.dataloader, desc="Training", dynamic_ncols=True)
    for step, prompt_text in enumerate(pbar):
      metrics = self.train_step(prompt_text)
      pbar.set_postfix({
          "policy_loss": f"{metrics['policy_loss']:.4f}",
          "reward_mean": f"{metrics['reward_mean']:.2f}", # reward 简写为 r
          "kl": f"{metrics['kl']:.4f}",
          "ratio":f"{metrics['ratio']:.4f}"
      })
    
  
  def log_prob(self,model,outputs,input_ids,attention_mask,grad=False):
    """
    将response的概率返回 [B*G, resp_len]
    

    :param model: 模型
    :param outputs: input+response
    :param input_ids: input的id
    :param attention_mask: mask，除response和left_padding均为1
    """
    
    outputs = outputs.to(model.device)
    attention_mask = attention_mask.to(model.device)
    if grad:
      logits = model(outputs, attention_mask=attention_mask).logits[:,input_ids.shape[1]-1:-1,:]
    else:
      with torch.no_grad():
        logits = model(outputs, attention_mask=attention_mask).logits[:,input_ids.shape[1]-1:-1,:]
    label = outputs[:,input_ids.shape[1]:]
    
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    log_prob = torch.gather(log_prob, dim=-1, index=label.unsqueeze(-1)).squeeze(-1)
    
    return log_prob
  def train_step(self, prompt_text):
    
    inputs,answers = self.prepare_data(prompt_text)
    model_answers,actor_log_prob,outputs,input_ids,attention_mask,resp_mask = self.rollout(inputs)
    reward,raw_reward = self.compute_reward(answers,model_answers)
    # print(f"奖励为：{raw_reward}")
    advantage = reward.unsqueeze(-1)
    # print(f"奖励为：{reward.shape}")
      
    ref_log_prob = self.log_prob(self.ref,outputs,input_ids,attention_mask)
    for _ in range(self.cfg.grpo.epoch):
      
      # print(f"真实答案为：{answers}")
      # print(f"模型答案为：{model_answers}")

      
      actor_new_log_prob = self.log_prob(self.actor,outputs,input_ids,attention_mask,grad=True)
      # 计算kl散度
      # r = ref_log_prob / actor_log_prob
      ref_log_prob = ref_log_prob.to(actor_new_log_prob.device)
      r = torch.exp(ref_log_prob - actor_new_log_prob)
      kl_3 = r-1-torch.log(r)
      
      ratio = torch.exp(actor_new_log_prob - actor_log_prob)
      surr1 = ratio * advantage
      swanlab_ratio = (ratio * resp_mask).sum() / (resp_mask.sum()+1e-8)
      surr2 = torch.clamp(ratio, 1-self.cfg.grpo.clip_ratio, 1+self.cfg.grpo.clip_ratio) * advantage
      policy_grad_loss = -torch.min(surr1, surr2)
      
      actor_loss = (policy_grad_loss * resp_mask).sum() / (resp_mask.sum()+1e-8)
      
      kl_loss = (kl_3 * resp_mask).sum() / (resp_mask.sum()+1e-8)
      loss = actor_loss + self.cfg.grpo.beta * kl_loss
      self.actor_optimizer.zero_grad()
      loss.backward()
      self.actor_optimizer.step()
      # print(f"loss为：{loss.item()}")
      
    metrics = {
              "policy_loss": actor_loss.item(),
              # "train/total_loss": (crit_loss + act_loss).item(),
              "reward_mean": sum(raw_reward)/len(raw_reward),
              "kl": kl_loss.mean().item(), # 这里的 kl 是你 compute_reward 里算的那个
              "ratio": swanlab_ratio.mean().item(),
          }
    if self.log:
      swanlab.log(metrics)
    return metrics
    
  def prepare_data(self, prompt_text):
    """_summary_
    数据预处理
    Args:
        prompt_text (_type_): 纯文本，包含问题和回答
    Returns:
        返回inputs和answers，前groupe_size个为一组
        inputs: [B * G, seq_len]
        answers: [B * G]
    """
    
    question_input = self.tokenizer(prompt_text["question"], padding=True, return_tensors="pt").to(self.cfg.base.device)
    
    answer = [GSM8KRatingUtils.extract_gold_answer(prompt_text["answer"][i])  for i in range(len(prompt_text)) for j in  range(self.cfg.grpo.group_size)]

    print(question_input["input_ids"].shape)
    print(question_input["attention_mask"].shape)
    print("重复batch")
    # repeat_interleave 会逐个重复，repeat会按组重复
    question_input["input_ids"] = torch.repeat_interleave(question_input["input_ids"], self.cfg.grpo.group_size, dim=0)
    question_input["attention_mask"] = torch.repeat_interleave(question_input["attention_mask"], self.cfg.grpo.group_size, dim=0)

    # print(question_input.shape)
    return question_input, answer
    
  def rollout(self, inputs):
    """_summary_
    生成一组数据
    Args:
        inputs (_type_): _description_
    Returns:dict
        answers: 模型的回答，取了answer部分
        actor_log_prob: 模型生成的概率
        input_ids: 输入的id
        attention_mask: 输入的mask，长度为seqlen，1包括input+有效resp
        resp_mask: 模型生成的mask，只包含有效回答的masks
        
    """
    
    outputs = self.actor.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=self.cfg.infer.max_new_tokens,
    temperature=self.cfg.infer.temperature,
    do_sample=True,
    top_p = self.cfg.infer.top_p,
    repetition_penalty = self.cfg.infer.repetition_penalty,
    top_k = self.cfg.infer.top_k,
    )
    answers_text = self.tokenizer.batch_decode(outputs[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
    # for i in range(len(answers_text)):
    #   print(f"模型输出为：{answers_text[i][-15:]}")
    answers = [GSM8KRatingUtils.extract_model_answer(answers_text[i]) for i in range(len(answers_text))]
    attention_mask = torch.zeros(outputs.shape, dtype=torch.long).to(self.cfg.base.device)
    attention_mask = (outputs != self.tokenizer.pad_token_id).long().to(self.cfg.base.device)

    # 这个mask，仅包含了有效回答，即只有输出有效token的位置为1，其余均为0，因为batch对齐的影响
    resp_mask = (outputs != self.tokenizer.pad_token_id).int()[:,inputs.input_ids.shape[1]:]
    # resp_mask[:, :inputs.input_ids.shape[1]] = 0
    # 重要性采样的logits [batch, seq_len-1, vocab_size]
    
    log_prob = self.log_prob(self.actor, outputs, inputs.input_ids, attention_mask)

    # return{
    #   "answers": answers,
    #   # 这个和inputs.input_ids一样，但是这个是toke
    #   "input_ids": inputs.input_ids,
    #   "attention_mask": attention_mask,
    #   "actor_log_prob": log_prob,
    # }
    return answers, log_prob, outputs,inputs.input_ids, attention_mask,resp_mask
  
  def compute_reward(self, answers, model_answers):
    """_summary_

    Args:
        answers (_type_): _description_
        model_answers (_type_): _description_
        
        
    """
    raw_reward = []
    for i in range(0,len(answers)):
      if not model_answers[i]:
        raw_reward.append(-5)
      else:
        res = GSM8KRatingUtils.check_correctness(answers[i], model_answers[i])
        if res:
          raw_reward.append(5)
        else:
          raw_reward.append(2)
    reward = torch.tensor(raw_reward).to(self.cfg.base.device).to(torch.float32)
    for group_ix in range(0,len(answers),self.cfg.grpo.group_size):
      mean = torch.mean(reward[group_ix:group_ix+self.cfg.grpo.group_size])
      # torch.std() 函数计算样本标准差，/n-1
      std = torch.std(reward[group_ix:group_ix+self.cfg.grpo.group_size])
      # 除以0，会报错，所以加1e-8
      reward[group_ix:group_ix+self.cfg.grpo.group_size] = (reward[group_ix:group_ix+self.cfg.grpo.group_size]-mean)/(std+ 1e-8)
    return reward,raw_reward
    
if __name__ == '__main__':
  grpo = GRPO(log=True)
  grpo.train()