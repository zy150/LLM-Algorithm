import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    """
    args:
        solution_str:回答字符串
        只匹配回答的最后300个字符
        建议使用默认的strict模式

        return answer

    """
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    计算奖励分数
    Args:
        solution_str: 模型回答
        ground_truth: 真实答案
        method: 解析答案的方式, choices are 'strict' and 'flexible'
        format_score: 格式正确的分数
        score: 答案正确的分数
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    # print(f"模型答案：{answer} Ground truth: {ground_truth}")
    if answer is None:
        return -0.3
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


def trl_reward_fn(prompts: list[str], completions: list[str], solution: list[str], **kwargs):
    """
    TRL库自定义的奖励函数，注意所有字段均为列表输入

    :param prompts: 数据的prompt输入
    :param completions: 模型回答
    :param solution: 答案
    :param kwargs: 说明
    return: score列表
    """
    rewards = []
    for model_answer, gt in zip(completions, solution):
        score = compute_score(model_answer, gt, "strict", 0.3, 1.0)

        rewards.append(score)
    # print(f"模型答案：{completions}")
    # print(f"输入：{prompts}")

    return rewards
