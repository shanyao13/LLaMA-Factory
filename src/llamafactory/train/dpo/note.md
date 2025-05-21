📌 一、DPO 训练整体逻辑
DPO (Direct Preference Optimization) 是一种强化学习方法，用于训练模型在两种回答中更倾向人类偏好的那一个。核心流程如下：

🧠 1. 数据格式
每个样本包含：

prompt：用户输入的提示；

chosen：人类偏好的回复；

rejected：不推荐的回复。

常见数据格式（jsonl）如下：

{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "France has many cities."
}
⚙️ 2. Tokenization
在数据加载阶段，prompt + chosen 和 prompt + rejected 分别拼接后 tokenize，例如：

# 拼接后输入
chosen_input = tokenizer(prompt + chosen, ...)
rejected_input = tokenizer(prompt + rejected, ...)
🧪 3. 模型前向计算
对于每个 batch，我们会构造两个输入（chosen 和 rejected）：

input_ids_chosen, attention_mask_chosen

input_ids_rejected, attention_mask_rejected

然后通过模型分别算出：

log_probs_chosen

log_probs_rejected

🎯 4. 损失计算（DPO loss）
核心损失函数（简化版）：

loss = -log(sigmoid(beta * (log_probs_chosen - log_probs_rejected)))
beta 是可调超参数，控制惩罚力度；

如果 chosen 明显优于 rejected，差值大，loss 小；

模型被训练为更倾向 chosen 的输出。

📦 二、精简版纯文本 DPO Collator
针对纯文本任务（无图像、视频、音频），你可以使用如下的 简洁版 DPO Collator，仅处理文本输入：

python
复制
编辑
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class TextOnlyDPODataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    padding: bool = True
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 分别提取 prompt + chosen 和 prompt + rejected
        prompts = [f["prompt"] for f in features]
        chosen_responses = [f["chosen"] for f in features]
        rejected_responses = [f["rejected"] for f in features]

        # 拼接：prompt + chosen/rejected
        chosen_texts = [p + c for p, c in zip(prompts, chosen_responses)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected_responses)]

        # 分别 tokenize
        chosen_batch = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors,
        )
        rejected_batch = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors,
        )

        return {
            "input_ids_chosen": chosen_batch["input_ids"],
            "attention_mask_chosen": chosen_batch["attention_mask"],
            "input_ids_rejected": rejected_batch["input_ids"],
            "attention_mask_rejected": rejected_batch["attention_mask"],
        }
✅ 三、使用方式（与 Trainer 配合）
配合 DPOTrainer 使用示例：

python
复制
编辑
from transformers import AutoTokenizer, TrainingArguments
from trl import DPOTrainer  # 或你自己实现的 DPOTrainer

tokenizer = AutoTokenizer.from_pretrained("your-model")

collator = TextOnlyDPODataCollator(tokenizer=tokenizer, max_length=1024)

trainer = DPOTrainer(
    model=your_model,
    args=TrainingArguments(...),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    beta=0.1,  # DPO loss 的超参数
)
trainer.train()
📌 四、总结：你当前场景推荐做法
组件	推荐
数据集格式	prompt, chosen, rejected
collator	使用上方 TextOnlyDPODataCollator
模型	支持 causal LM（如 LLaMA, GPT-2/3, Baichuan 等）
loss 函数	使用 DPO loss
多模态逻辑	可完全移除，不影响文本任务训练

