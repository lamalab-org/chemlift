import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import Dataset
import pandas as pd
from functools import partial
from typing import List
from tqdm import tqdm
from more_itertools import chunked

# models have different conventions for naming the attention modules
LORA_TARGET_MODULES_MAPPING = {
    "alpaca_native": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # llama-based -> decoder only
    "bigscience/bloom": ["query_key_value"],  # decoder only
    "bigscience/bloom-3b": ["query_key_value"],  # decoder only
    "microsoft/deberta-v2-xxlarge": ["query_proj", "value_proj"],
    "EleutherAI/gpt-neo-125m": ["q_proj", "v_proj"],
    "EleutherAI/gpt-neo-1.3B": ["q_proj", "v_proj"],
    "EleutherAI/gpt-neo-2.7B": ["q_proj", "v_proj"],
    "EleutherAI/gpt-neox-20b": ["query_key_value"],
    "EleutherAI/pythia-12b-deduped": ["query_key_value"],
    "EleutherAI/pythia-6.9b-deduped": ["query_key_value"],  # decoder only
    "EleutherAI/pythia-2.8b-deduped": ["query_key_value"],  # decoder only
    "EleutherAI/pythia-1.4b-deduped": ["query_key_value"],  # decoder only
    "EleutherAI/pythia-1b-deduped": ["query_key_value"],  # decoder only
    "EleutherAI/pythia-410m-deduped": ["query_key_value"],  # decoder only
    "EleutherAI/pythia-160m-deduped": ["query_key_value"],  # decoder only
    "EleutherAI/pythia-70m-deduped": ["query_key_value"],
    "gpt2": ["c_attn"],
    "EleutherAI/gpt-j-6b": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],  # decoder only
    "deepset/roberta-base-squad2": ["query", "value"],
    "t5-base": ["q", "v"],  # encoder - decoder
    "tiiuae/falcon-7b": ["query_key_value"],
}

PADDING_SIDE_MAPPING = {
    "alpaca_native": "left",
    "bigscience/bloom": "left",
    "bigscience/bloom-3b": "left",
    "gpt2": "left",
    "EleutherAI/gpt-neo-125m": "left",
    "EleutherAI/gpt-neo-1.3B": "left",
    "EleutherAI/gpt-neo-2.7B": "left",
    "EleutherAI/gpt-neox-20b": "left",
    "EleutherAI/pythia-12b-dedupedz``": "left",
    "EleutherAI/pythia-6.9b-deduped": "left",
    "EleutherAI/pythia-2.8b-deduped": "left",
    "EleutherAI/pythia-1.4b-deduped": "left",
    "EleutherAI/pythia-1b-deduped": "left",
    "EleutherAI/pythia-410m-deduped": "left",
    "EleutherAI/pythia-160m-deduped": "left",
    "EleutherAI/pythia-70m-deduped": "left",
    "EleutherAI/gpt-j-6b": "left",
    "llama": "left",
    "microsoft/deberta-v2-xxlarge": "right",
    "deepset/roberta-base-squad2": "right",
    "t5-base": "right",
    "opt": "left",
    "tiiuae/falcon-7b": "left",
}


def freeze(model):
    """Freeze the model."""
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model(
    base_model: str = "gptj", load_in_8bit: bool = True, lora_kwargs: dict = {}
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = PADDING_SIDE_MAPPING[base_model]

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map="sequential",
    )

    freeze(model)

    lora_default_kwargs = {
        "r": 16,  # lora attention size
        "lora_alpha": 32,  # "When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. As a result, we simply set α to the first r we try and do not tune it."
        "target_modules": LORA_TARGET_MODULES_MAPPING[base_model],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    lora_kwargs = {**lora_default_kwargs, **lora_kwargs}

    # now, apply LoRA to the model
    config = LoraConfig(
        **lora_kwargs,
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model, tokenizer


def tokenize(
    prompt,
    tokenizer,
    cutoff_len=1024,
    return_tensors=None,
    padding=False,
    truncation=True,
):
    result = tokenizer(
        prompt,
        truncation=truncation,
        max_length=cutoff_len,
        padding=padding,
        return_tensors=return_tensors,
    )
    return result


def tokenize_prompt(
    data_point,
    tokenizer,
    cutoff_len=1024,
    add_completion=True,
    return_tensors=None,
    padding=False,
    truncation=True,
):
    if add_completion:
        full_prompt = data_point["prompt"] + data_point["completion"]
    else:
        full_prompt = data_point["prompt"]
    tokenized_full_prompt = tokenize(
        full_prompt,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        return_tensors=return_tensors,
        padding=padding,
        truncation=truncation,
    )
    return tokenized_full_prompt


def train_model(
    model,
    tokenizer,
    train_data: pd.DataFrame,
    train_kwargs: dict = {},
    hub_model_name: str = None,
    report_to: str = None,
    tokenizer_kwargs: dict = {},
):
    default_train_kwargs = {
        "per_device_train_batch_size": 128,
        "warmup_steps": 100,
        "num_train_epochs": 20,
        "learning_rate": 3e-4,
        "fp16": True,
        "optim": "adamw_torch",
        "output_dir": "./output",
        "report_to": report_to,  # can be used for wandb tracking
    }
    train_kwargs = {**default_train_kwargs, **train_kwargs}
    tokenize_partial = partial(tokenize_prompt, tokenizer=tokenizer, **tokenizer_kwargs)
    train_data = Dataset.from_pandas(train_data).shuffle().map(tokenize_partial)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(**train_kwargs),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        trainer.train()

    if hub_model_name is not None:
        model.push_to_hub(hub_model_name)


def complete(
    model,
    tokenizer: AutoTokenizer,
    prompt_text: List[str],
    max_length: int = 50,
    top_k: int = 1,
    top_p: float = 0.9,
    temperature: float = 1.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.2,
    num_beams: int = 1,
    padding: bool = True,
    truncation: bool = True,
    batch_size: int = 64,
) -> List[dict]:
    model.eval()
    all_completions = []
    with torch.no_grad():
        for chunk in tqdm(
            chunked(range(len(prompt_text)), batch_size),
            total=len(prompt_text) // batch_size,
        ):
            batch = [prompt_text[i] for i in chunk]

            tokenize_partial = partial(
                tokenize,
                tokenizer=tokenizer,
                cutoff_len=1024,
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
            )
            prompt = tokenize_partial(batch)
            out = model.generate(
                inputs=prompt["input_ids"].to(model.device),
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                attention_mask=prompt["attention_mask"].to(model.device),
            )

            this_completion = [
                {"out": o, "decoded": tokenizer.decode(o, skip_special_tokens=True)}
                for o in out
            ]
            all_completions.extend(this_completion)

    return all_completions
