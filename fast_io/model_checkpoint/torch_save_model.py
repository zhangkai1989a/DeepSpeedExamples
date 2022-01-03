import time
import argparse
import torch
import os
from transformers import AutoModelForCausalLM
from transformers import T5ForConditionalGeneration
import deepspeed
from deepspeed.ops.aio import AsyncIOBuilder


AIO_QUEUE_DEPTH = 8
AIO_BLOCK_SIZE = 8*(1024**2)
AIO_THREAD_COUNT = 1
AIO_SINGLE_SUBMIT = False
AIO_OVERLAP_EVENTS = False
PINNED_BUFFER_MB = 64 

def _get_model(big_model):
    if big_model: 
        model_name="EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(model_name).half()#.cuda()
        ckpt_name="gpt-j-6B"
    else:
        model_name="hf-internal-testing/tiny-random-t5" # "patrickvonplaten/t5-tiny-random" # "t5-small"
        model = T5ForConditionalGeneration.from_pretrained(model_name).half()
        ckpt_name="t5-small"
    
    return model, model_name, ckpt_name


def _get_aio_handle():
    h = AsyncIOBuilder().load().aio_handle(
        block_size=AIO_BLOCK_SIZE,
        queue_depth=AIO_QUEUE_DEPTH,
        single_submit=AIO_SINGLE_SUBMIT,
        overlap_events=AIO_SINGLE_SUBMIT,
        num_threads=AIO_THREAD_COUNT)
    return h

def test_save(file, buffer, use_zipfile, io_buffer_mb):
    st = time.time()
    torch.save(f=file, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    return time.time() - st


def test_ds_mock_save(file, buffer, use_zipfile, io_buffer_mb):
    from deepspeed.io import MockFileWriter
    st = time.time()
    dsmw = MockFileWriter(file)
    torch.save(f=dsmw, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    write_sec = time.time() - st
    dsmw._dump_state()
    return write_sec 

def test_ds_py_save(file, buffer, use_zipfile, io_buffer_mb):
    from deepspeed.io import PyFileWriter
    st = time.time()
    dspw = PyFileWriter(file)   
    torch.save(f=dspw, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    write_sec = time.time() - st
    dspw._dump_state()
    return write_sec 

def test_ds_aio_save(file, buffer, use_zipfile, io_buffer_mb):
    h = _get_aio_handle()
    pinned_memory = torch.zeros(io_buffer_mb*(1024**2), dtype=torch.uint8, device='cpu').pin_memory()                                            
    from deepspeed.io import DeepSpeedFileWriter as dsfw
    st = time.time()
    dsfw = dsfw(
        file_path=file, 
        aio_handle=h,
        pinned_tensor=pinned_memory)
    torch.save(f=dsfw, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    write_sec = time.time() - st
    dsfw._dump_state()
    return write_sec

def run(model, model_name, ckpt_name, folder, legacy_save, io_buffer_mb):
    print(f'Model name = {model_name}')
    fn_dict = {
        'test_save': test_save, 
        'test_ds_mock_save': test_ds_mock_save, 
        'test_ds_py_save': test_ds_py_save,
        'test_ds_aio_save':test_ds_aio_save
    }
    for tag, fn in fn_dict.items():
        file = os.path.join(folder, f'{tag}_{ckpt_name}.pt')
        print(f'checkpoint file = {file}')
        if os.path.isfile(file):
            os.remove(file)
        st = time.time()
        write_sec = fn(file, model, not legacy_save, io_buffer_mb)
        ckpt_size = os.path.getsize(file)
        gb_size = ckpt_size/(1024**3)
        gb_per_sec = gb_size/write_sec
        print(f'{tag} -- {gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} gb/s')
        print(f'*********************************************')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        default=None,
                        type=str,
                        required=True,
                        help='Folder to use for I/O.')
    parser.add_argument('--big_model',
                        action='store_true',
                        help='Use EleutherAI/gpt-j-6B for checkpointing.')
    parser.add_argument('--legacy',
                        action='store_true',
                        help='Use torch legacy save format')
    
    parser.add_argument('--io_buffer_mb',
                        type=int,
                        default=PINNED_BUFFER_MB,
                        required=True,
                        help='Size of pinned i/o buffer in MB.')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def main():
    print(f'Performance test of deepspeed fast model checkpoint')
    print(f'torch version = {torch.__version__}')
    torch.manual_seed(42)
    args = parse_arguments()
    if not os.path.exists(args.folder):
        print(f'Invalid folder: {args.folder}')
        quit()
    model, model_name, ckpt_name = _get_model(args.big_model)
    
    run(model, model_name, ckpt_name, args.folder, args.legacy, args.io_buffer_mb)
    

if __name__ == "__main__":
    main()
