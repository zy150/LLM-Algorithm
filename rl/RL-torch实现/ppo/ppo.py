from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import check_gpu_memory
import time
from torch import nn

import swanlab


@dataclass
class BaseConfig:
    device: str = 'cuda:4'
    dtype: str = 'bfloat16'
    actor_model_path: str = '/models/Qwen/Qwen3-0.6B'
    critic_model_path: str = '/models/Qwen/Qwen3-0.6B'


@dataclass
class RewardConfig():
    device: str = 'cuda:2'
    dtype: str = 'bfloat16'
    model_path: str = "/models/reward/internlm/internlm2-1_8b-reward"


@dataclass
class InferConfig(BaseConfig):
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class PPOConfig:
    learning_rate: float = 5e-6  # 5e-6 ～ 1e-5
    batch_size: int = 2
    kl_coef: float = 0.1
    clip_eps: float = 0.2
    gamma: float = 0.99
    lambd: float = 0.95

    def __post_init__(self):
        self.ppo_epoch = 6
        self.update_timestep = 1000
        self.minibatch_size = self.batch_size * self.update_timestep


@dataclass
class MainConfig:
    # 将不同模块的配置嵌套在一起
    base: BaseConfig = field(default_factory=BaseConfig)
    infer: InferConfig = field(default_factory=InferConfig, metadata={
                               'help': 'infer config'})
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 42


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        # 在这里只处理好模板字符串，不要转成 ID 否则dataloader会长度不一致报错
        self.formatted_prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "你是一个人工智能助手。"},
                    {"role": "user",   "content": p}
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]

    def __len__(self):
        return len(self.formatted_prompts)

    def __getitem__(self, ix):
        return self.formatted_prompts[ix]  # 返回的是字符串


def getRewardPrompt(responses):
    """  
        构造奖励模型所需的格式

        {"role": "system", "content": "你是一个人工智能助手。"},
        {"role": "user", "content": "Hello! What's your name?"}, 
        {"role": "assistant", "content": resp}

        responses: 模型生成的结果，纯字符串，仅回复
    """
    return [
        [
            # system也会影响分数
            {"role": "system", "content": "你是一个人工智能助手。"},
            {"role": "user", "content": "Hello! What's your name?"},
            {"role": "assistant", "content": resp}
        ]
        for resp in responses
    ]


class CriticModel(nn.Module):
    def __init__(self, model_path, device, dtype):
        super().__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),  # 利用字符串取属性
            trust_remote_code=True
        )

        # 获取隐藏层维度
        hidden_size = self.base_model.config.hidden_size

        # 定义 Value Head: 线性映射 [hidden_size -> 1]
        # 这一层需要随机初始化
        self.value_head = nn.Linear(
            hidden_size, 1, device=device, dtype=getattr(torch, dtype))
        self.to(device)

    def forward(self, prompt):
        # 获取隐藏状态 这里不调用generate 只forward产生logits
        outputs = self.base_model(
            **prompt,
            output_hidden_states=True
        )
        # outputs.logits
        # outputs.hidden_states 当为 True 时，返回所有层的隐藏状态
        # 拿到最后一层的隐藏状态: [batch, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # 经过 Value Head -> [batch, seq_len, 1]
        values = self.value_head(last_hidden_state)

        # 压缩维度 -> [batch, seq_len]
        # 每一个 Token 位置都会有一个预估分数
        return values.squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, model_path, device, dtype):
        super().__init__()

        self.base_model = AutoModel.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),  # 利用字符串取属性
            trust_remote_code=True
        )
        self.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

    def get_score(self, text_list):
        """
        get_score 的 Docstring

        :param text: share GPT格式的文本,多个batch以列表存储
        """

        score = self.base_model.get_score(self.tokenizer, text_list)

        return score


def init_ppo_models(cfg: MainConfig):
    device = cfg.base.device
    dtype = getattr(torch, cfg.base.dtype)

    actor = AutoModelForCausalLM.from_pretrained(
        cfg.base.actor_model_path, dtype=dtype, trust_remote_code=True
    ).to(device)
    actor.train()

    ref = AutoModelForCausalLM.from_pretrained(
        cfg.base.actor_model_path, dtype=dtype, trust_remote_code=True
    ).to(device)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad = False  # 冻结参数

    critic = CriticModel(cfg.base.critic_model_path, device, cfg.base.dtype)
    critic.train()  # 开启训练模式

    reward_model = RewardModel(
        cfg.reward.model_path, cfg.reward.device, cfg.reward.dtype)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    check_gpu_memory(device)
    return actor, ref, critic, reward_model


class PPO:
    def __init__(self, prompts):
        # 加载模型
        actor, ref, critic, reward_model = init_ppo_models(cfg)
        self.actor = actor
        self.ref = ref
        self.critic = critic
        self.reward_model = reward_model

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base.actor_model_path, padding_side='left')
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.ppo.learning_rate)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=cfg.ppo.learning_rate)

        self.dataloader = DataLoader(PromptDataset(
            prompts, self.tokenizer), batch_size=cfg.ppo.batch_size)

        self.cfg = cfg

        swanlab.login(api_key="xxx", save=True)
        # https://docs.swanlab.cn/api/py-init.html
        swanlab.init(
            project="LLM-RL",
            config=cfg.__dict__,
            experiment_name="ppo-Qwen3-0.6B",
            description="ppo练习，纯torch实现"
        )

    def train(self):

        for prompt_text in self.dataloader:

            self.train_step(prompt_text)

    def train_step(self, prompt_text=[]):
        """核心步骤：
        1. 采样 (Rollout): 让模型说话，记录发生的每一件事
        2. 计算奖励: 结合 Reward Model 的分数和 KL 惩罚
        3. PPO 更新: 根据优势函数更新模型参数
        """
        # 采样 (Rollout)
        experience = self.collect_experience(prompt_text)
        # 计算奖励
        rewards, just_resp_mask, kl = self.compute_reward(experience)
        # 优势和Q
        advantages, returns = self.compute_advantage(
            experience, rewards, just_resp_mask)

        for _ in range(cfg.ppo.ppo_epoch):
            input_len = experience["inputs"]["input_ids"].shape[1]
            # PPO 训练
            new_logits = self.actor(**experience["all_inputs"]).logits
            labels = experience["all_inputs"]["input_ids"][:, input_len:]
            all_logprobs = torch.nn.functional.log_softmax(
                new_logits, dim=-1)[:, input_len-1:-1]
            new_logprobs = torch.gather(
                all_logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            # print(f"new_log shape: {new_logprobs.shape}")
            # print(f"act_log shape: {experience['actor_logprobs'].shape}")
            ratio = torch.exp(new_logprobs - experience["actor_logprobs"])
            # min(ratio*A, clip(ratio)*A)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.ppo.clip_eps,
                                1.0 + cfg.ppo.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2)
            # --Critcic--
            new_values = self.critic(experience["all_inputs"])[
                :, input_len-1:-1]

            # 计算 Critic Loss (MSE)
            # 公式：(new_values - returns)^2
            value_loss = (new_values - returns) ** 2

            # 总 Loss = Actor_Loss + 系数 * Value_Loss
            # 用 mask 抹掉 Padding 区域，然后求平均
            # total_loss = (policy_loss + value_loss) * just_resp_mask
            # loss = total_loss.sum() / just_resp_mask.sum()

            # 计算最终的 Actor 损失（标量）
            act_loss = (policy_loss * just_resp_mask).sum() / \
                just_resp_mask.sum()
            self.actor_optimizer.zero_grad()
            act_loss.backward()
            self.actor_optimizer.step()

            # 计算最终的 Critic 损失（标量）
            crit_loss = (value_loss * just_resp_mask).sum() / \
                just_resp_mask.sum()
            self.critic_optimizer.zero_grad()
            crit_loss.backward()
            self.critic_optimizer.step()

        swanlab.log({
            "train/policy_loss": act_loss.item(),
            "train/value_loss": crit_loss.item(),
            # "train/total_loss": (crit_loss + act_loss).item(),
            "reward_mean": rewards.mean().item(),
            "kl": kl.mean().item(),  # 这里的 kl 是你 compute_reward 里算的那个
            "train/ratio": ratio.mean().item(),
        })

    def collect_experience(self, prompt_text):
        """
        采样，获取actor模型的generate和prob

        :param prompt_text: input文本
        """

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.cfg.base.device)
        # [batch, seq_len]

        """  
            stop_strings: 停止符,遇到某个字符串时，就停止生成
            use_cache:是否使用kv缓存，如果为True，则使用缓存，否则不使用缓存
            
            temperature:默认1.0 大于1，缩小 logits除以T
            top_p:默认1.0 把词表里的词按概率从高到低排序，然后一个一个往下累加，直到总和达到 top_p 为止。剩下的词全部扔掉 1为不开启
            top_k:默认50 表示只从概率最高的前 50 个词里选
            num_beams: beam search的 beam size，默认为1
            num_beam_groups:将束分为几组，会进行多样心惩罚，观察前面组的一些词
            repetition_penalty: 重复惩罚，默认1.0,为1表示无惩罚
            min_p:如果最高词概率是 0.8，min_p 设为 0.1，那么只有概率大于 0.8*0.1=0.08 的词才能留
            do_sample:只有开了采样，top那些参数才会生效
            
            都开启：
            Logtis-> T -> min-p -> top-k -> top-p -> softmax(这里是剩余的重新softmax使其概率和为1)
            其中top-k是硬筛选，top-p是软筛选，剩余概率不足p直接进入下一步
        """
        outputs = self.actor.generate(
            **inputs,
            # 生成长度不包括prompt，max_length包括，都设置了，后者被覆盖
            max_new_tokens=cfg.infer.max_new_tokens,
            # min_length和min_new_tokens同理

            temperature=cfg.infer.temperature,
            do_sample=True,
            top_p=cfg.infer.top_p,
            repetition_penalty=cfg.infer.repetition_penalty,
        )

        input_length = inputs['input_ids'].shape[1]
        responses = [self.tokenizer.decode(
            output[input_length:], skip_special_tokens=True) for output in outputs]
        # critic
        # 只要不是 pad_token 的地方，都是 1 outpupts只有token，没有mask
        # print(f"type outputs != self.tokenizer.pad_token {type(outputs != self.tokenizer.pad_token)}")
        # 忽略padding batch推理时会出现补齐现象
        full_mask = (outputs != self.tokenizer.pad_token_id).int()
        # print(f"full_mask.shape : {full_mask.shape}")
        # print(f"full_mask : {torch.nonzero(full_mask[0])}")
        # print(outputs[0][-3:-1])
        critic_inputs = {
            "input_ids": outputs,
            "attention_mask": full_mask  # 没有任何end
        }
        # bs,len(outputs(input+response)) mask为0的地方，value并不为0，可能因为有bias？只是注意力不计算，loss一定要频闭
        values = self.critic(critic_inputs)
        print(f"values.shape : {values.shape}")
        print(f"full_mask.shape : {full_mask.shape}")
        # 需要错位因为是St的时候，做了动作，判断出的的Vt+1，而最后一个状态似乎没用
        last_token_indices = (torch.arange(
            values.shape[1], device=values.device) * full_mask).max(dim=1).values

        # for i in range(values.shape[0]):
        #     start_pos = last_token_indices[i]
        #     print(f"第 {i} 行从末尾开始的内容: {values[i, start_pos:]}")
        """  
            不为0，因为即使有mask，只是注意力不看它们而已，还是会输出，有bias layernorm什么的
        """
        values = values[:, input_length-1:-1]
        # print(f"responses.shape : {responses.shape}")

        with torch.no_grad():
            # 拿到全序列的 logits
            """
                这里会把所有词的下一个概率都计算了，因为GPU并行，算就算了，虽然没啥用，概率是整个词表的概率
                而且调用forward只会算一次，即预测一个词，generate会重复迭代，直到出现eos 或者 max_length
            """
            logits = self.actor(outputs, attention_mask=full_mask).logits

            # --- 关键：计算选中 Token 的 Logprobs ---
            # logits 预测的是下一个词，所以要错位对齐
            # [batch, seq_len-1, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous()
            # [batch, seq_len-1]
            shift_labels = outputs[:, 1:].contiguous()

            # 计算 log_softmax 对softmax取对数 防止数字过大
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # 抠出模型选中的那个词的概率 gether函数，取index 的元素，gather函数的dim参数指定了索引的维度
            # [bs, seq, vocab_size] 变成了 [bs, seq, 1] -> [bs, seq]
            all_logprobs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # 截取 Response 部分的 logprobs
            # 因为 shift 后，索引 input_length-1 对应的就是 response 的第一个 token
            actor_logprobs = all_logprobs[:, input_length-1:]
        return {
            "inputs": inputs,
            "outputs": outputs,
            "responses": responses,
            "input_length": input_length,
            "actor_logprobs": actor_logprobs,
            "values": values,
            # 输入inputs+outputs的字典
            "all_inputs": critic_inputs,
        }

    def compute_reward(self, experience):
        """
        计算奖励 rt + Rt - KLs
        :param experience: 采样得到的参数

        返回的维度为 [bs, seq_len] 由于是批次采样，seq_len可能会有大量eos
        """
        responses = experience["responses"]
        full_mask = experience["all_inputs"]["attention_mask"]
        with torch.no_grad():
            # 不是inputs是outputs
            ref_outputs = self.ref(**experience["all_inputs"])
            # bs，seqlen，vocabsize  不要最后一个预测词的概率
            shift_logits = ref_outputs.logits[:, :-1, :].contiguous()
            # 与第i个词的概率对齐shift_logits[i]为预测第i个词的logits，shift_labels[i]为第i个词的label
            shift_labels = experience["outputs"][:, 1:].contiguous()
            ref_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            # ref_probs bs,seq,vocabsize
            ref_log_probs = torch.gather(
                ref_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)[:, experience["input_length"]-1:]
        # 计算 KL 惩罚 (Token-wise)
        kl = experience["actor_logprobs"] - ref_log_probs
        token_rewards = - self.cfg.ppo.kl_coef * kl

        # 获取 Reward Model 的分值 (对整个 sequence) 这里是非Token-level Reward 加到最后一个token上
        # rm_score [batch_size] 也可以token-level
        rm_scores = self.reward_model.get_score(
            getRewardPrompt(responses))  # [batch_size]
        rm_scores = torch.tensor(
            rm_scores,
            device=token_rewards.device,
            dtype=token_rewards.dtype
        )
        # token_rewards[:, -1]+= rm_scores dim等于谁就把谁消除
        last_indices = (full_mask * torch.arange(full_mask.size(1),
                        device=token_rewards.device)).max(dim=1).values
        input_len_offset = 1+experience["input_length"] - 1
        rel_last_indices = last_indices - input_len_offset

        batch_size, resp_len = token_rewards.shape
        # 注意，我们指定了device，因为是Tensor运算，不可以跨设备，因为创建Tensor默认在CPU如果不指定device，只有Tensor*1.0这样的标量可以不指定
        # 利用广播，[1,seq_len] <= [batch_size, 1] 即每行小于索引的全部为1，大于为0，变成[batch_size, seq_len]
        mask = torch.arange(resp_len, device=token_rewards.device).unsqueeze(
            0) <= rel_last_indices.unsqueeze(1)
        # 这里的mask是bool，必须float一下，也可以用mask.to(token_rewards.dtype)
        # 关于to，可以device=，dtype=，还可以copy什么的，设备用字符串，类型用torch.float32
        token_rewards = token_rewards * mask.float()  # 抹除 PAD 区域的 KL

        row_idx = torch.arange(full_mask.size(0))
        # 这么写会把每行的索引都取出来
        # print(token_rewards[:,rel_last_indices])
        # print(token_rewards[row_idx,rel_last_indices])

        # 4. 把 RM 分数加到最后一个有效的 token 奖励上
        # 找到每个 sequence 真正的结束位置，加上 rm_scores
        token_rewards[row_idx, rel_last_indices] += rm_scores
        return token_rewards, mask, kl  # 返回有效的token长度mask，这里不包括eos，最后一个即位有效字符最后一个,
        # [batch, response_len]

    def compute_advantage(self, experience, rewards, just_resp_mask):
        """
        计算GAE优势和Returns

        :param rewards: [batch, resp_len]
        :param values: [batch, resp_len]
        :param just_resp_mask: [batch, resp_len] 只有有效字符的输出mask 无input长度

        return advantages, returns
        """

        batch_size, resp_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        values = experience["values"]  # 注意，也没有input长度
        # 我们需要保存一个“后一时刻”的 GAE 值，初始化为 0
        last_gae_lam = 0

        gamma = self.cfg.ppo.gamma
        lambd = self.cfg.ppo.lambd

        for t in reversed(range(resp_len)):
            # 选中所有行，取第t+1列
            next_values = values[:, t + 1] if t < resp_len - 1 else 0
            # TD-Error = rt + gamma*Vt+1 - Vt.  如果这一位是 Padding，delta 应该是 0
            delta = (rewards[:, t] + gamma * next_values -
                     values[:, t]) * just_resp_mask[:, t]
            # 计算GAE 如果这一位是有效字符，At = delta + 衰减后的未来优势
            # A_t = delta + gamma * lambda * last_gae_lam
            At = delta + gamma * lambd * last_gae_lam * just_resp_mask[:, t]
            last_gae_lam = At
            advantages[:, t] = At
        """ 
            returns实际是Q，是给Critic用的，让它学习
            这个不用标准化，我们的目的就是让critic预测分数，假如标准化了，就失去了绝对分数的感知
        """
        returns = advantages + values
        # 优势是给actor用的，标准化能使训练更稳定
        advantages = self.standardize_advantages(advantages, just_resp_mask)
        # detach是指移除梯度，不进行反向传播。即从计算图剥离，从模型算过来的值会带有计算图信息
        # 我们做了许多操作，如果不detach，计算图会一直记录操作，以备计算梯度，占用显存
        return advantages.detach(), returns.detach()

    def standardize_advantages(self, advantages, mask):
        """
        只针对有效区域进行标准化
        :param advantages: [batch, resp_len]
        :param mask: [batch, resp_len] (1是有效, 0是padding)
        """
        num_valid = mask.sum()

        # 计算有效区域的均值
        mean = (advantages * mask).sum() / num_valid

        # 公式: std = sqrt( E[(x-mean)^2] )
        # 在计算平方差时，依然要用 mask 把 padding 部分抹掉
        sq_diff = ((advantages - mean) ** 2) * mask
        variance = sq_diff.sum() / num_valid
        std = torch.sqrt(variance + 1e-8)  # 加个极小值防止除以0

        # 减去均值，除以标准差
        advantages = (advantages - mean) / std

        # 因为减去均值后，原本 0 的位置会变成 -mean/std，必须抹掉
        advantages = advantages * mask

        return advantages


if __name__ == '__main__':
    prompt_list = [
        # --- 逻辑与数学 ---
        "如果 3 只猫在 3 分钟内能抓 3 只老鼠，那么 100 只猫抓 100 只老鼠需要多少分钟？",
        "请证明为什么 0.999...（循环）等于 1。",
        "桌子上有 5 支点燃的蜡烛，风吹灭了 2 支，最后还剩几支蜡烛？",
        "如果你在比赛中超过了第二名，你是第几名？",

        # --- 创意写作与角色扮演 ---
        "写一个关于一架拥有自我意识的钢琴的短篇故事，字数 200 字左右。",
        "你现在是一名为太空游客服务的厨师，请向我推荐三道‘星际特色菜’。",
        "用鲁迅的文风写一段关于现代人刷短视频的评论。",
        "创作一首关于‘程序员的深夜’的现代诗。",

        # --- 知识与科普 ---
        "简单解释一下什么是量子纠缠，让一个 10 岁的孩子也能听懂。",
        "为什么天空是蓝色的，但在日落时会变红？",
        "如果地球停止自转 1 秒钟会发生什么？",
        "请对比一下人工智能中的‘监督学习’和‘强化学习’的区别。",

        # --- 营销与职场 (对应你之前的营销专家需求) ---
        "为一家新开的猫咪咖啡馆写 5 条吸引年轻人的小红书文案标题。",
        "如何礼貌地拒绝老板在周六晚上提出的加班要求？写一封邮件草稿。",
        "作为营销专家，请分析为什么‘盲盒’这种商业模式会让人上瘾？",
        "为一个智能水壶写一个 30 秒的短视频脚本，突出它的‘自动泡茶’功能。",

        # --- 编程与技术 ---
        "用 Python 写一个快速排序算法，并添加详细注释。",
        "什么是 RESTful API？它的核心原则有哪些？",
        "如何解释 Docker 容器和虚拟机的区别？",
        "写一个正则表达式，用于验证电子邮箱地址的合法性。",

        # --- 伦理与开放式思考 ---
        "你认为人工智能拥有情感吗？为什么？",
        "如果你能回到过去改变一件历史事件，你会选哪件？为什么？",
        "为什么人们喜欢在水族馆里游泳，而不是在游泳池里？（重新测试这条）",
        "如果一台机器可以完美模拟你的意识，那么那个模拟物是你吗？",

        # --- 实用工具与格式化 ---
        "将以下内容总结为三个要点：[此处插入任意长文本]",
        "请将‘人工智能将改变世界’这句话翻译成法语、日语和德语。",
        "帮我制定一份为期一周的‘低碳水减脂’食谱。",
        "如何在一个月内学会弹奏尤克里里？请给出详细计划。",

        # --- 陷阱与幽默 ---
        "周杰伦的《双截棍》里，一共踢了多少次腿？",
        "请用最严肃的语气解释‘为什么煎饼果子是圆的’。",
    ]
    cfg = MainConfig()
    ppo = PPO(prompt_list)

    ppo.train()
