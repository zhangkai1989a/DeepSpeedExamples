import time
import torch
from torch.optim import Adam
import os
from torch_save_utils import test_save, test_ds_mock_save, test_ds_py_save, test_ds_fast_save
from save_model_utils import get_model, validate_arguments, parse_arguments


def run(model, model_name, ckpt_name, folder, legacy_save, io_buffer_mb,
        show_statistics):
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
        write_sec = fn(file, model, not legacy_save, io_buffer_mb,
                       show_statistics)
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


def main():
    print(
        f'Performance test of torch.save() integration of fast model checkpointing.'
    )
    print(f'torch version = {torch.__version__}')
    torch.manual_seed(42)

    args = parse_arguments()
    if not validate_arguments(args):
        quit()

    model, model_name, ckpt_name = get_model(args.model)
    if args.half:
        model = model.half()
    if args.gpu:
        model = model.cuda()
    if args.optimizer:
        optimizer = _get_initialized_optimizer(model, args.fused)
        ckpt_state = {'model': model, 'optimizer': optimizer}
    else:
        ckpt_state = {'model': model}
    run(ckpt_state, model_name, ckpt_name, args.folder, args.legacy,
        args.io_buffer_mb, not args.no_statistics)


if __name__ == "__main__":
    main()
