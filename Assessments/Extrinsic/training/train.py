import os
import torch
import contextlib
import transformers
from typing import Literal
from utils import common, data_utils
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field


@dataclass
class ModelArguments :
    model_name_or_path : str = field(default = None)
    attn_implementation : Literal["eager", "sdpa", "flash_attention_2"] = field(default = "eager")
    use_lora : bool = field(default = True)
    lora_rank : int = field(default = 256)
    lora_alpha : float = field(default = 512)

@dataclass
class DataArguments :
    dataset_type : Literal["Finetune", "SFT"] = field(default = "SFT")
    SFT_format : Literal["mathematics"] = field(default = "mathematics")
    dataset_path : str = field(default = None)
    eval_size : int = field(default = 1)

@dataclass
class TrainingArguments(transformers.TrainingArguments) :
    model_max_length : int = field(default = None)
    initialize_model_on_cpu : bool = field(default = False)
    wandb_project : str = field(default = "EvalTree")
    lora_dropout : float = field(default = 0.1)
    model_max_length : int = field(default = 1024)


if __name__ == "__main__" :
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if training_args.deepspeed is not None :
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    elif training_args.initialize_model_on_cpu :
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = True
    else :
        ctx_mgr = common.staggered_object_creation(local_rank = training_args.local_rank, world_size = training_args.world_size)
        device_map = {"" : training_args.device.index}
        low_cpu_mem_usage = True
    def load_model() -> transformers.PreTrainedModel :
        with ctx_mgr :
            torch_dtype = torch.float32
            assert not (training_args.fp16 and training_args.bf16), "Cannot use both fp16 and bf16 at the same time."
            if training_args.fp16 :
                torch_dtype = torch.float16
            elif training_args.bf16 :
                torch_dtype = torch.bfloat16
            else :
                pass
            model : transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path, token = os.getenv("HF_TOKEN"),
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
                attn_implementation = model_args.attn_implementation,
                low_cpu_mem_usage = low_cpu_mem_usage,
                device_map = device_map,
                torch_dtype = torch_dtype,
            )
            if model_args.use_lora :
                peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM, inference_mode = False, r = model_args.lora_rank, lora_alpha = model_args.lora_alpha, lora_dropout = training_args.lora_dropout, target_modules = ["q_proj", "v_proj", "output_proj"])
                model = get_peft_model(model, peft_config)
            else :
                raise NotImplementedError
        return model
    model = load_model()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, token = os.getenv("HF_TOKEN"),
        model_max_length = training_args.model_max_length,
        padding_side = "right",
    )
    assert tokenizer.eos_token is not None, "Tokenizer must have an EOS token."
    special_tokens_dict = dict()
    if tokenizer.pad_token is None :
        special_tokens_dict["pad_token"] = "[PAD]"
    common.stable_resize_token_embeddings_and_tokenizer(model, tokenizer, special_tokens_dict)

    data_module : dict = data_utils.make_data_module(
        tokenizer = tokenizer,
        data_args = data_args,
        training_args = training_args,
    )

    if data_args.dataset_type in ("Finetune", "SFT") :
        trainer = transformers.Trainer(
            model = model,
            tokenizer = tokenizer,
            args = training_args,
            **data_module,
        )
    else :
        raise NotImplementedError
    trainer.train()
    common.warning("Training is done.", main_process_only = True)
    trainer.save_model()
    common.warning("The model is saved.", main_process_only = True)
    trainer.save_state()
    common.warning("The trainer state is saved.", main_process_only = True)