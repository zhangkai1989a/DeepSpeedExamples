import argparse
import os
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
from torch_save_utils import PINNED_BUFFER_MB


def _get_gpt_j_6B(tag):
    model_name = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ckpt_name = "gpt-j-6B"
    return model, model_name, ckpt_name


def _get_tiny_t5(tag):
    model_name = "hf-internal-testing/tiny-random-t5"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ckpt_name = "tiny-random-t5"
    return model, model_name, ckpt_name


def _get_hf_gpt2(tag):
    model_name = tag
    model = AutoModelForCausalLM.from_pretrained(tag)
    ckpt_name = tag
    return model, model_name, ckpt_name


HF_MODELS = {
    'tiny-t5': _get_tiny_t5,
    'gpt-j-6B': _get_gpt_j_6B,
    'gpt2': _get_hf_gpt2,
    'gpt2-large': _get_hf_gpt2,
    'gpt2-xl': _get_hf_gpt2,
}


def get_model(model_tag):
    return HF_MODELS[model_tag](model_tag)


def validate_arguments(args):
    success = True
    if not os.path.exists(args.folder):
        print(f'Invalid folder: {args.folder}')
        success = False

    if not args.model in HF_MODELS:
        print(f'{args.model} is not a supported HF model tag')
        success = False

    if args.optimizer and args.half:
        if not args.gpu:
            print(f'mixed precision only supported with gpu tensors')
            success = False

    return success


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

    parser.add_argument('--half',
                        action='store_true',
                        help='Use half-precision tensors.')

    parser.add_argument(
        '--io_buffer_mb',
        type=int,
        default=PINNED_BUFFER_MB,
        help=f'Size of pinned i/o buffer in MB. Default = {PINNED_BUFFER_MB}')

    parser.add_argument('--zero_stage',
                        type=int,
                        default=0,
                        help='ZeRO optimization stage. Default = 0')

    parser.add_argument('--cpu_offload',
                        action='store_true',
                        help='Enable CPU offload of optimizer state.')

    parser.add_argument('--no-statistics',
                        action='store_true',
                        help='Suppress low-level performance statistics.')

    parser.add_argument('--single_io_buffer',
                        action='store_true',
                        help='Disable double buffering of i/o buffer.')

    args = parser.parse_args()
    print(f'args = {args}')
    return args
