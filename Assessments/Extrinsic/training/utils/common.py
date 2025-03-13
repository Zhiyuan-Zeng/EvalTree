import os
import torch
import logging
import transformers


def stable_resize_token_embeddings_and_tokenizer(
    model : transformers.PreTrainedModel,
    tokenizer : transformers.PreTrainedTokenizer,
    special_tokens_dict : dict,
) :
    """Resize tokenizer and embedding together.

    For new tokens, the embedding value is the average of all old embedding vectors.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    stable_resize_token_embeddings(model, len(tokenizer))

def stable_resize_token_embeddings(model : transformers.PreTrainedModel, target_size : int, jitter_new_embeddings = False) :
    num_new_tokens = target_size - model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(target_size)

    if num_new_tokens > 0:
        @torch.inference_mode()
        def stable_init(embedding) :
            embedding_data = embedding.weight.data
            embedding_avg = embedding_data[: -num_new_tokens].mean(dim = 0, keepdim = True)
            embedding_data[-num_new_tokens :] = embedding_avg
            if jitter_new_embeddings:
                embedding_std = embedding_data[: -num_new_tokens].std(dim = 0, keepdim = True)
                # The random tensor must be of the same shape as the new embeddings.
                embedding_data[-num_new_tokens :] += torch.randn_like(embedding_data[-num_new_tokens :]) * embedding_std

        input_embeddings = model.get_input_embeddings()  # Must grab this again after resize.
        output_embeddings = model.get_output_embeddings()
        # It doesn't matter if there's weight sharing or not; with sharing, the second init will overwrite the first.
        for embeddings in (input_embeddings, output_embeddings) :
            stable_init(embeddings)


class staggered_object_creation(object) :
    """
    Objection creation in a distributed setting could be very RAM-intensive.

    This function staggers the creation of objects on odd and even ranks, so that not all objects
    are created at once.

    Assumes local_rank == -1 means no distributed training.
    """

    def __init__(self, local_rank : int, world_size : int) :
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size

    def __enter__(self, *args, **kwargs) :
        del args, kwargs
        if self.world_size > 1 and self.local_rank % 2 == 0 :
            torch.distributed.barrier()
        return self

    def __exit__(self, *args, **kwargs) :
        del args, kwargs
        if self.world_size > 1 :
            if self.local_rank % 2 == 1 :
                torch.distributed.barrier()
            torch.distributed.barrier()  # Final safety barrier.

    def __call__(self, func) :
        def decorator(*args, **kwargs) :
            with self :
                return func(*args, **kwargs)
        return decorator


def get_local_rank() :
    return int(os.getenv("LOCAL_RANK", 0))

def is_main_process():
    """Return True if the current process is the main process."""
    return get_local_rank() <= 0


logger = logging.getLogger(__name__)
def warning(msg, main_process_only = True, *args, **kwargs) :
    if (not main_process_only) or is_main_process() :
        logger.warning(msg, *args, **kwargs)