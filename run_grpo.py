# accelerate launch --num_processes 4 run_grpo.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve --model qgallouedec/SmolLM2-360M-Rickified --data_parallel_size 4 --max_model_len 1024

from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import re


def format_reward(completions, **kwargs):
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def correctness_reward(completions, solutions, **kwargs):
    rewards = []
    for completion, ground_truths in zip(completions, solutions):
        content = completion[0]["content"]
        matches = [ground_truth in content for ground_truth in ground_truths]
        reward = 1.0 if any(matches) else 0.0
        rewards.append(reward)
    return rewards


def train():
    dataset = load_dataset("qgallouedec/rick-physics-grpo", split="train")

    def format_dataset(example):
        return {"prompt": [{"role": "user", "content": example["question"]}]}

    dataset = dataset.map(format_dataset)

    args = GRPOConfig(
        max_completion_length=512,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        num_generations=16,
        mask_truncated_completions=True,
        # Speedup and reduce memory
        gradient_checkpointing=True,
        bf16=True,
        use_vllm=True,
        output_dir="data/SmolLM2-360M-Rickified-GRPO",
        # Logging
        run_name="SmolLM2-360M-Rickified-GRPO",
        logging_steps=2,
        log_completions=True,
        num_completions_to_print=1,
    )

    trainer = GRPOTrainer(
        model="qgallouedec/SmolLM2-360M-Rickified",
        reward_funcs=[format_reward, correctness_reward],
        train_dataset=dataset,
        args=args,
    )
    trainer.train()
    trainer.push_to_hub(dataset_name="qgallouedec/rick-physics-grpo")


if __name__ == "__main__":
    train()
