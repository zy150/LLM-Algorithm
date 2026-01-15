from datasets import load_dataset
from utils import exarct_answer
from reward_score import compute_score, extract_solution, trl_reward_fn


def get_gsm8k_dataset(text_len: int = 500, split: str = "train"):
    dataset = load_dataset("openai/gsm8k", "main", split=split,
                           token="your-token")

    dataset = dataset.filter(lambda x: len(x["question"]) < text_len)

    return dataset


def make_map_fn(split):

    def process_fn(example, idx):

        question = example.pop("question")

        answer = example.pop("answer")

        number = extract_solution(answer)

        prompt = f"""
        A conversation between User and Assistant.
        The user asks a question, and the Assistant solves it.
        The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        Then, the assistant response with '####' followed by the final answer.\n
        User: {question}. \n Assistant: Let's think step by step.
        """
        prompt_instruct = f"""
        Given the following math problem, please solve it step by step.
        Format your response as follows:
        1. Begin with your reasoning processes.
        2. End your response with '#### ' followed strictly by the numerical answer only. 

        Constraints:
        - Do NOT use LaTeX formatting (like \\boxed, $$, etc.) for the final answer.
        - Do NOT add text like "The answer is" after '####'.
        - Just output the number (integer or float).

        The question: {question}
        """
        data = {
            "type": split,
            "data_source": "gsm8k",
            "question": question,
            "prompt_instruct": [
                {
                    "role": "user",
                    "content": prompt_instruct
                },

            ],
            "prompt": prompt,
            "solution": number
        }
        return data
    return process_fn


def get_dataset(text_len: int = 500, split: str = "train"):
    """
    get_dataset 的 Docstring

    :param text_len: 过滤问题长度大于text_len的数据，限制在text_len以内
    :type text_len: int
    :param split: 取训练集或者测试集 train/test
    :type split: str
    """
    dataset = get_gsm8k_dataset(text_len, split)
    # split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    # train_dataset = split_dataset["train"]
    # eval_dataset = split_dataset["test"]
    dataset = dataset.map(
        function=make_map_fn(split), with_indices=True)

    return dataset


if __name__ == "__main__":
    dataset = get_dataset(text_len=500, split="train")
    print(dataset[0])
    print(dataset)
