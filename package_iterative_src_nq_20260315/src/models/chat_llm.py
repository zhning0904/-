import os
import json
import hashlib
import torch
import logging
from pathlib import Path
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional during pure base-model inference
    PeftModel = None

logging.getLogger("transformers").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROLES_PATH = PROJECT_ROOT / "experiments" / "model_roles.json"
DEFAULT_MODELSCOPE_MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
DEFAULT_MODELSCOPE_CACHE_DIR = PROJECT_ROOT / "modelscope-cache"
DEFAULT_LLM_RESPONSE_CACHE_PATH = PROJECT_ROOT / "outputs" / "cache" / "chat_llm_response_cache.json"


def _resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def _load_model_roles() -> dict:
    # 角色配置缺失时返回空字典，由调用方走默认路径
    if not MODEL_ROLES_PATH.exists():
        return {}
    with MODEL_ROLES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


class ChatLLM:
    """
    Generic chat LLM wrapper (ModelScope/HF-compatible).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODELSCOPE_MODEL_ID,
        adapter_path: str | None = None,
        device_map: str | dict | None = "auto",
        torch_dtype: torch.dtype | str = torch.float16,
        max_memory: dict | None = None,
        offload_folder: str | None = None,
        cache_dir: str | None = None,
        system_prompt: str | None = None,  # 默认 system prompt
        enable_response_cache: bool = True,
        response_cache_path: str | None = None,
    ):
        # 统一限定模型缓存目录到项目根目录下，避免写入系统根分区
        if cache_dir is None:
            cache_dir = str(DEFAULT_MODELSCOPE_CACHE_DIR)
        self.cache_dir = cache_dir

        # 确保缓存目录存在且位于 /root/autodl-tmp 下
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            # 即使创建失败也不终止，后续 ModelScope 可能自行处理
            pass

        self.model_path = model_path
        self.adapter_path = adapter_path
        self.enable_response_cache = bool(enable_response_cache)
        if response_cache_path is None:
            response_cache_path = str(DEFAULT_LLM_RESPONSE_CACHE_PATH)
        self.response_cache_path = _resolve_path(response_cache_path)
        self._response_cache: dict[str, dict] = {}
        if self.enable_response_cache:
            self._load_response_cache()

        # Fallback for environments where CUDA exists but is unusable for this torch build.
        # Keep defaults for healthy GPU setups; only override when no usable CUDA is detected.
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            # CPU 回退：避免在无可用 CUDA 时继续使用 fp16 触发算子报错
            if device_map in (None, "auto"):
                device_map = "cpu"
            if torch_dtype in (torch.float16, "float16", "fp16"):
                torch_dtype = torch.float32

        # 如果传入的是 ModelScope 模型 ID（而非本地已存在路径），
        # 使用 snapshot_download 显式下载到项目内的缓存目录。
        mp_path = Path(model_path)
        if not mp_path.exists():
            local_dir = snapshot_download(model_path, cache_dir=self.cache_dir)
            self.model_path = local_dir
            model_path = local_dir
        else:
            self.model_path = str(mp_path.resolve())

        if adapter_path:
            adapter_resolved = _resolve_path(adapter_path)
            if not Path(adapter_resolved).exists():
                raise FileNotFoundError(f"Adapter path not found: {adapter_resolved}")
            self.adapter_path = adapter_resolved

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
        )

        if self.adapter_path:
            if PeftModel is None:
                raise RuntimeError("peft is required to load a LoRA adapter, but it is not installed.")
            # rewrite 角色：base model + LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path).eval()
        else:
            # general 角色：直接使用基座模型
            self.model = base_model.eval()

        self.system_prompt = system_prompt

    def _make_response_cache_key(self, payload: dict) -> str:
        """Build a stable cache key for request payloads."""
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_response_cache(self) -> None:
        """Load persistent response cache from disk."""
        cache_file = Path(self.response_cache_path)
        if not cache_file.exists():
            self._response_cache = {}
            return
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            self._response_cache = {}
            return

        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        if isinstance(entries, dict):
            self._response_cache = {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}
        else:
            self._response_cache = {}

    def _save_response_cache(self) -> None:
        """Persist response cache to disk."""
        cache_file = Path(self.response_cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": self._response_cache}
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

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
    # 内部：只做一次 template + tokenize + 上卡
    # ============================================================
    @torch.no_grad()
    def _prepare_inputs(self, messages: list[dict]):
        """
        返回:
        - inputs: dict(tensor...) on model.device
        - input_len: prompt token length
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(prompt, return_tensors="pt")
        # BatchEncoding 支持 .to(device)，但 dict 显式搬运更稳妥
        inputs = {k: v.to(self.model.device) for k, v in enc.items()}
        input_len = inputs["input_ids"].shape[-1]
        return inputs, input_len

    # ============================================================
    # 单次 chat 调用（对外行为不变）
    # ============================================================
    @torch.no_grad()
    def call_chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        cache_key = ""
        if self.enable_response_cache:
            cache_key = self._make_response_cache_key(
                {
                    "mode": "call_chat",
                    "model_path": self.model_path,
                    "adapter_path": self.adapter_path,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                }
            )
            hit = self._response_cache.get(cache_key)
            if hit is not None and isinstance(hit.get("text"), str):
                return hit["text"]

        # ✅ 优化点：避免在 call_chat 内部重复构造/上卡（但仍保持接口完全不变）
        inputs, input_len = self._prepare_inputs(messages)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        if temperature == 0:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        outputs = self.model.generate(**gen_kwargs)
        gen = outputs[0][input_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        if self.enable_response_cache and cache_key:
            self._response_cache[cache_key] = {"text": text}
            self._save_response_cache()

        return text

    # ============================================================
    # CoRAG-style 采样（保持原调用方式与机制：1 greedy + N-1 sampled）
    # ============================================================
    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ):
        """
        对齐 CoRAG best-of-N：
        - 第 1 条 greedy
        - 其余 N-1 条采样
        - 保持原调用方式不变

        速度优化（不改变底层生成机制）：
        1) prompt/template/tokenize/to(device) 只做一次
        2) N-1 条采样使用 num_return_sequences 批量生成（每条仍是独立采样）
        """
        assert n >= 1, "n must be >= 1"

        cache_key = ""
        if self.enable_response_cache:
            cache_key = self._make_response_cache_key(
                {
                    "mode": "sample",
                    "model_path": self.model_path,
                    "adapter_path": self.adapter_path,
                    "prompt": prompt,
                    "n": int(n),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "top_p": float(top_p),
                    "system_prompt": system_prompt if system_prompt is not None else self.system_prompt,
                }
            )
            hit = self._response_cache.get(cache_key)
            if hit is not None and isinstance(hit.get("outputs"), list):
                return [str(x) for x in hit.get("outputs", [])]

        messages = self.build_messages(prompt, system_prompt)

        # ✅ 优化点1：只做一次 tokenize + 上卡
        inputs, input_len = self._prepare_inputs(messages)

        common_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        outputs_text: list[str] = []

        # 1) greedy：等价于原来 i==0 时 temp=0 -> do_sample=False
        greedy_out = self.model.generate(
            **common_kwargs,
            do_sample=False,
        )
        greedy_gen = greedy_out[0][input_len:]
        outputs_text.append(self.tokenizer.decode(greedy_gen, skip_special_tokens=True).strip())

        if n == 1:
            return outputs_text

        # 2) sampled：等价于原来 i>0 的 do_sample=True, temperature/top_p
        # 只是合并为一次批量生成，不改变采样分布/机制
        sampled_out = self.model.generate(
            **common_kwargs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n - 1,
        )

        for seq in sampled_out:
            gen = seq[input_len:]
            outputs_text.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())

        if self.enable_response_cache and cache_key:
            self._response_cache[cache_key] = {"outputs": outputs_text}
            self._save_response_cache()

        return outputs_text


# ================== 使用示例 ==================
if __name__ == "__main__":

    llm = ChatLLM(
        system_prompt="你是一个查询重写专家。只输出改写后的查询。"
    )

    user_query = "who plays killer croc in the movie suicide squad"

    # 直接传 prompt，不用自己写 messages
    candidates = llm.sample(
        prompt=f"Rewrite the query to improve retrieval:\n{user_query}",
        n=10,
        temperature=0.7,
        max_tokens=64,
    )

    for i, c in enumerate(candidates):
        print(f"[{i}] {c}")


# Backward compatibility: keep old class name for legacy scripts/notebooks
QwenChatLLM = ChatLLM


def create_rewrite_llm(**kwargs) -> ChatLLM:
    """Create rewrite model from model_roles.json.

    Supports two modes:
    1) base + LoRA adapter (when adapter_path is set)
    2) base-only rewrite model (when adapter_path is null/empty)
    """
    roles = _load_model_roles()
    rewrite = roles.get("rewrite", {})
    model_path = str(rewrite.get("base_model_path", DEFAULT_MODELSCOPE_MODEL_ID))
    adapter_path = rewrite.get("adapter_path")
    system_prompt = rewrite.get("system_prompt")

    return ChatLLM(
        model_path=model_path,
        adapter_path=str(adapter_path) if adapter_path else None,
        system_prompt=system_prompt,
        **kwargs,
    )


def create_general_llm(**kwargs) -> ChatLLM:
    """Create general-purpose model without rewrite adapter."""
    roles = _load_model_roles()
    general = roles.get("general", {})
    model_path = str(general.get("base_model_path", DEFAULT_MODELSCOPE_MODEL_ID))
    system_prompt = general.get("system_prompt")

    return ChatLLM(
        model_path=model_path,
        adapter_path=None,
        system_prompt=system_prompt,
        **kwargs,
    )