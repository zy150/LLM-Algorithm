from swanlab.integration.transformers import SwanLabCallback
from data_proccess import get_dataset
import swanlab
from reward_score import trl_reward_fn
from trl import GRPOConfig, GRPOTrainer
import os
from transformers import AutoTokenizer, TrainerCallback


def get_training_args(model_type):
    tranning_args = GRPOConfig(
        output_dir=save_path(model_type, type="gsm8K-strict"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        use_vllm=False,  # 为True时，需要自己启动vllm，端口8000
        weight_decay=0.01,
        warmup_ratio=0.1,
        optim="adamw_8bit",
        report_to=["swanlab"],
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=False,
        num_generations=8,  # group size
        generation_batch_size=8,  # 一次生成一组的多少个 此参数必须和num_generations一致
        loss_type='grpo',
        max_steps=500,
        epsilon=0.2,
        learning_rate=1e-6,
        beta=0.001,  # 默认是0 省显存
        epsilon_high=0.28,  # one sided
        # SFTTrainer 有max_length这个，GRPO没有
        max_prompt_length=200,      # 输入（Prompt）的最大长度
        max_completion_length=1200,  # 输出的最大生成长度
        # eval_strategy="steps",  # 传了评估数据的话，这里要设置
        # eval_steps=10,
    )

    return tranning_args


def get_trainer(model, reward_fn, train_dataset, eval_dataset, args):
    """
    获取trainer对象

    :param model: 模型路径
    :param reward_fn: 自定义的奖励函数
    :param train_dataset: 训练集
    :param eval_dataset: 可以自己定义评估集，目前代码未启用评估
    :param args: tranning_args
    """
    swan_cb = SwanLabCallback(
        project="Qwen0.6B-Base-GRPO-Gsm8k",
        experiment_name="grpo-gsm8k数据集复现",
        description=""
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,  # 此参数并非必须，但如果是指令对话模型，必须传，会自动调用apply_chat_template
        callbacks=[swan_cb],

    )

    return trainer


def save_path(model, path="./output", type="gsm8k"):

    return f"{path}/{type}/{model}"


if __name__ == "__main__":
    train_dataset = get_dataset()
    args = get_training_args("base")

    model = "/models/Qwen/Qwen3-0.6B-Base"
    # model = "/models/Qwen/Qwen3-0.6B"
    trainer = get_trainer(model, trl_reward_fn,
                          train_dataset, None, args)

    trainer.train()

    trainer.save_model()
