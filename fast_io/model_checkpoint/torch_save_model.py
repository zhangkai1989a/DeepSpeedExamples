import time
import argparse
import torch
from torch.optim import Adam
import os
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from torch_save_utils import PINNED_BUFFER_MB
from torch_save_utils import test_save, test_ds_mock_save, test_ds_py_save, test_ds_fast_save


def _get_gpt_j_6B(tag):
    model_name = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)  #.half()
    ckpt_name = "gpt-j-6B"
    return model, model_name, ckpt_name


def _get_tiny_t5(tag):
    model_name = "hf-internal-testing/tiny-random-t5"
    model = T5ForConditionalGeneration.from_pretrained(model_name)  #.half()
    ckpt_name = "tiny-random-t5"
    return model, model_name, ckpt_name


def _get_hf_gpt2(tag):
    model_name = tag
    model = AutoModelForCausalLM.from_pretrained(tag)
    ckpt_name = tag
    return model, model_name, ckpt_name


HF_MODELS = {
    'tiny-random-t5': _get_tiny_t5,
    'gpt-j-6B': _get_gpt_j_6B,
    'gpt2': _get_hf_gpt2,
    'gpt2-large': _get_hf_gpt2,
    'gpt2-xl': _get_hf_gpt2,
}


def _get_model(model_tag):
    return HF_MODELS[model_tag](model_tag)


def run(model, model_name, ckpt_name, folder, legacy_save, io_buffer_mb):
    print(f'Model name = {model_name}')
    fn_dict = {
        'test_save': test_save,
        'test_ds_mock_save': test_ds_mock_save,
        'test_ds_py_save': test_ds_py_save,
        'test_ds_fast_save': test_ds_fast_save
    }
    for tag, fn in fn_dict.items():
        file = os.path.join(folder, f'{tag}_{ckpt_name}.pt')
        print(f'checkpoint file = {file}')
        if os.path.isfile(file):
            os.remove(file)
        st = time.time()
        write_sec = fn(file, model, not legacy_save, io_buffer_mb)
        ckpt_size = os.path.getsize(file)
        gb_size = ckpt_size / (1024**3)
        gb_per_sec = gb_size / write_sec
        print(
            f'{tag} -- {gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} gb/s'
        )
        print(f'*********************************************')


def _get_initialized_optimizer(model, fused_opt):
    base_optimizer = Adam(model.parameters())
    import deepspeed
    if fused_opt:
        from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer as FP16_Wrapper
    else:
        from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer as FP16_Wrapper
    optimizer = FP16_Wrapper(base_optimizer)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    return optimizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        default=None,
                        type=str,
                        required=True,
                        help='Folder to use for I/O.')

    parser.add_argument(
        '--model',
        default=None,
        type=str,
        required=True,
        help='Hugging Face transformers tag of model (e.g., gpt2).')

    parser.add_argument('--legacy',
                        action='store_true',
                        help='Use torch legacy save format')

    parser.add_argument('--optimizer',
                        action='store_true',
                        help='Include optimizer state in checkpoint.')

    parser.add_argument('--fused',
                        action='store_true',
                        help='Use fused fp16 optimizer.')

    parser.add_argument('--gpu', action='store_true', help='Use gpu tensors.')

    parser.add_argument('--io_buffer_mb',
                        type=int,
                        default=PINNED_BUFFER_MB,
                        help='Size of pinned i/o buffer in MB.')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def validate_arguments(args):
    success = True
    if not os.path.exists(args.folder):
        print(f'Invalid folder: {args.folder}')
        success = False

    if not args.model in HF_MODELS:
        print(f'{args.model} is not a supported HF model tag')
        success = False

    return success


def main():
    print(f'Performance test of deepspeed fast model checkpoint')
    print(f'torch version = {torch.__version__}')
    torch.manual_seed(42)

    args = parse_arguments()
    if not validate_arguments(args):
        quit()

    model, model_name, ckpt_name = _get_model(args.model)
    if args.optimizer:
        model = model.half().cuda()
        optimizer = _get_initialized_optimizer(model, args.fused)
        ckpt_state = {'model': model, 'optimizer': optimizer}
    else:
        model = model.half()
        if args.gpu:
            model = model.cuda()
        ckpt_state = {'model': model}
    run(ckpt_state, model_name, ckpt_name, args.folder, args.legacy,
        args.io_buffer_mb)


if __name__ == "__main__":
    main()
