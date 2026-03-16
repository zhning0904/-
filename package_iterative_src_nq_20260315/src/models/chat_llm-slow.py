import os
import torch
import logging
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

logging.getLogger("transformers").setLevel(logging.ERROR)


class QwenChatLLM:
    """
    Qwen-7B-Chat agent
    """

    def __init__(
        self,
        model_path: str = "qwen/Qwen-7B-Chat",
        device_map: str | dict | None = "auto",
        torch_dtype: torch.dtype | str = torch.float16,
        max_memory: dict | None = None,
        offload_folder: str | None = None,
        cache_dir: str | None = None,
        system_prompt: str | None = None,  # 默认 system prompt
    ):
        if cache_dir is not None and ("/" in model_path) and os.path.sep not in model_path:
            model_path = snapshot_download(model_path, cache_dir=cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
        ).eval()

        self.system_prompt = system_prompt

    # ============================================================
    # 自动包装 messages
    # ============================================================
    def build_messages(self, prompt: str, system_prompt: str | None = None):
        """
        根据 prompt 自动构造 chat messages
        """
        messages = []

        sp = system_prompt if system_prompt is not None else self.system_prompt
        if sp:
            messages.append({"role": "system", "content": sp})

        messages.append({"role": "user", "content": prompt})
        return messages

    # ============================================================
    # 单次 chat 调用
    # ============================================================
    @torch.no_grad()
    def call_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        if temperature == 0:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen = outputs[0][input_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ============================================================
    # CoRAG-style 采样
    # ============================================================
    def sample(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.9,
        max_tokens: int = 256,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ):
        """
        对齐 CoRAG best-of-N：
        - 第 1 条 greedy
        - 其余 N-1 条采样
        - 每条独立 forward（不是 num_return_sequences）
        """

        messages = self.build_messages(prompt, system_prompt)

        outputs = []
        for i in range(n):
            temp = 0.0 if i == 0 else temperature
            outputs.append(
                self.call_chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                )
            )

        return outputs


# ================== 使用示例 ==================
if __name__ == "__main__":

    llm = QwenChatLLM(
        system_prompt="你是一个查询重写专家。只输出改写后的查询。"
    )

    user_query = "who plays killer croc in the movie suicide squad"

    # 直接传 prompt，不用自己写 messages
    candidates = llm.sample(
        prompt=f"Rewrite the query to improve retrieval:\n{user_query}",
        n=5,
        temperature=0.7,
        max_tokens=64,
    )

    for i, c in enumerate(candidates):
        print(f"[{i}] {c}")