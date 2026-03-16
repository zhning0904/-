import sys
import os
import logging
from modelscope import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import MODEL_PATH
import torch

# MODEL_PATH = "qwen/Qwen-7B-Chat"
logging.getLogger("transformers").setLevel(logging.ERROR)


class ChatLLM:
    """A minimal ChatGPT-style LLM wrapper for local Qwen models."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device_map: str | dict | None = "auto",
        torch_dtype: torch.dtype | str = torch.float16,
        max_memory: dict | None = None,
        offload_folder: str | None = None,
    ):
        """初始化模型与分词器。

        Args:
            model_path: ModelScope 模型名称或本地模型路径。
            device_map: 传给 transformers `device_map`，可用 "auto" / "cpu" / {"cuda:0": ...}。
            torch_dtype: 模型精度，CPU 上推荐 torch.float32。
            max_memory: 可选，限制每个设备的显存上限，格式同 transformers。
            offload_folder: 可选，启用 cpu/offload 时存放权重的目录。
        """

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 加载模型，并自动分配到可用设备；支持 CPU/offload 降低显存占用
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
        ).eval()  # 进入推理模式（关闭dropout等）


    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 5,
        do_sample: bool = True,
    ):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        results = []
        for output in outputs:
            generated_tokens = output[input_len:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(text.strip())

        return results if n > 1 else results[0]



# ================== 测试入口 ==================
if __name__ == "__main__":
    # 实例化模型
    llm = ChatLLM()
    # 测试 prompt
    prompt = "法国的首都是哪座城市?"
    # 调用生成方法
    result = llm.generate(prompt)
    print("生成结果:", result)
