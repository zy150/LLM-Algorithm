import torch


def check_gpu_memory(device_id: str):
    """
    查看指定 GPU 的显存占用情况
    :param device_id: 字符串，如 'cuda:4'
    """
    device_index = torch.cuda._utils._get_device_index(device_id)

    # 总显存（单位 Byte -> GB）
    total_mem = torch.cuda.get_device_properties(
        device_index).total_memory / 1024**3

    # 已分配 / 预留 / 历史峰值
    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**3

    # 计算百分比
    usage_ratio = allocated / total_mem * 100

    print(f"\n--- GPU 显存监控 ({device_id}) ---")
    print(f"总显存 (Total):       {total_mem:.2f} GB")
    print(f"已分配 (Allocated):   {allocated:.2f} GB  ({usage_ratio:.1f}%)")
    print(f"已预留 (Reserved):    {reserved:.2f} GB")
    print(f"历史峰值 (Max):       {max_allocated:.2f} GB")
    print("-------------------------------\n")
