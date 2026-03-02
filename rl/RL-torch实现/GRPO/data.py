from datasets import load_dataset

# gsm8k = load_dataset("openai/gsm8k","main",split="train")
# gsm8k = gsm8k.filter(lambda x: len(x["question"]) < 400)
# print(gsm8k[0])


import re

class GSM8KRatingUtils:
    @staticmethod
    def extract_gold_answer(text: str):
        """
        从 GSM8K 数据集的 'answer' 字段中提取标准答案。
        GSM8K 的标准格式通常是："... reasoning ... #### 42"
        """
        # 找到 #### 之后的所有内容
        parts = text.split("####")
        if len(parts) < 2:
            return None # 数据格式异常
        
        # 取最后一部分，并去掉首尾空格
        answer_str = parts[-1].strip()
        
        # 清理格式：去掉逗号(1,000 -> 1000)，去掉货币符号($5 -> 5)
        # 注意：这里保留了小数点和负号
        clean_answer = answer_str.replace(",", "").replace("$", "").replace(" ", "")
        return clean_answer

    @staticmethod
    def extract_model_answer(text):
        """
        从模型输出中提取纯数字答案。
        逻辑：查找 <answer> 标签内容 -> 提取最后一个数字 -> 找不到则返回 None。
        """
        if not text or not isinstance(text, str):
            return None
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
        if not answer_match:
            return None

        content = answer_match.group(1)
        
        # 从标签内容中找数字
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)

        # 只有存在数字时才返回最后一个，否则返回 None
        if numbers:
            return numbers[-1]
        
        return None


    @staticmethod
    def check_correctness(gold_str, model_str):
        """
        判断模型答案是否正确。
        """
        if gold_str is None or model_str is None:
            return False
            
        # 尝试数值对比 (处理 42.0 == 42 的情况)
        try:
            gold_num = float(gold_str)
            model_num = float(model_str)
            # 允许极小的浮点误差
            return abs(gold_num - model_num) < 1e-5
        except ValueError:
            # 如果转不了数字（比如答案是 "Monday"），则回退到字符串精确匹配
            return gold_str == model_str
