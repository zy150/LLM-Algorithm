import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from data_proccess import get_dataset
from reward_score import compute_score, extract_solution, trl_reward_fn
from transformers import AutoTokenizer


def main(model_type="base"):
    """
        model_type:指定prompt的类型，如果为base，则使用prompt字段
                    如果是非base，使用prompt_instruct，此为指令对话版本

    """
    OUTPUT_FILE = f"./output/eval_results_{model_type}.jsonl"
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        dtype="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
    )

    test_dataset = get_dataset(split="test")

    prompts = []
    metadata = []  # 存 ground_truth 等信息
    correct_count = 0
    has_answer = 0
    for item in test_dataset:

        if model_type == "base":
            prompt_content = item["prompt"]
        else:
            prompt_content = item["prompt_instruct"]

        prompts.append(prompt_content)
        metadata.append({
            "ground_truth": item["solution"],
            "question": item.get("question", ""),
        })
    formatted_prompts = [
        tokenizer.apply_chat_template(
            p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    if model_type != "base":
        prompts = formatted_prompts

    outputs = llm.generate(prompts, sampling_params)

    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            prompt_used = prompts[i]

            final_response = generated_text
            score = trl_reward_fn(prompt_used, [final_response], [
                                  metadata[i]["ground_truth"]])[0]

            if score == 1:
                correct_count += 1
                print(f"✅ {i}")
            elif score == 0.3:
                has_answer += 1  # 统计有答案的数量
            item = {
                "question": metadata[i]["question"],
                "model_response": final_response,  # 完整的回复
                "ground_truth": metadata[i]["ground_truth"],
                "correct": score == 1,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        has_answer += correct_count
        statics = {
            "accuracy": correct_count / len(test_dataset),
            # 有答案的题目里，对的有多少
            "answer_accuracy": correct_count / (has_answer + 1e-5),
            "total": len(test_dataset),
            "answer_total": has_answer,
            "answer_acc_count": correct_count,
            # 整体准确率
            "Accuracy": correct_count / len(test_dataset)
        }
        f.write(json.dumps(statics, ensure_ascii=False) + "\n")
    print(f"整体准确率: {correct_count / len(test_dataset):.4f}")
    print(f"有答案的题目里，对的有: {correct_count / (has_answer + 1e-5):.4f}")


if __name__ == "__main__":
    MODEL_PATH = "./output/gsm8K-strict/Instrcut"   # 你的模型路径
    # MODEL_PATH = "/models/Qwen/Qwen3-0.6B"

    main("instruct")
