import time
import argparse
import torch
import os
import deepspeed
from deepspeed.ops.aio import AsyncIOBuilder


AIO_QUEUE_DEPTH = 8
AIO_BLOCK_SIZE = 8*(1024**2)
AIO_THREAD_COUNT = 1
AIO_SINGLE_SUBMIT = False
AIO_OVERLAP_EVENTS = False
PINNED_BUFFER_MB = 64 


def _get_aio_handle():
    h = AsyncIOBuilder().load().aio_handle(
        block_size=AIO_BLOCK_SIZE,
        queue_depth=AIO_QUEUE_DEPTH,
        single_submit=AIO_SINGLE_SUBMIT,
        overlap_events=AIO_SINGLE_SUBMIT,
        num_threads=AIO_THREAD_COUNT)
    return h

def test_save(file, buffer, use_zipfile):
    st = time.time()
    torch.save(f=file, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    return time.time() - st


def test_ds_mock_save(file, buffer, use_zipfile):
    from deepspeed.io import MockFileWriter
    st = time.time()
    dsmw = MockFileWriter(file)
    torch.save(f=dsmw, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    write_sec = time.time() - st
    dsmw._dump_state()
    return write_sec 

def test_ds_py_save(file, buffer, use_zipfile):
    from deepspeed.io import PyFileWriter
    st = time.time()
    dspw = PyFileWriter(file)   
    torch.save(f=dspw, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    write_sec = time.time() - st
    dspw._dump_state()
    return write_sec 

def test_ds_aio_save(file, buffer, use_zipfile):
    h = _get_aio_handle()
    pinned_memory = torch.zeros(PINNED_BUFFER_MB*(1024**2), dtype=torch.uint8, device='cpu').pin_memory()                                            
    from deepspeed.io import DeepSpeedFileWriter as dsfw
    st = time.time()
    dsfw = dsfw(
        file_path=file, 
        aio_handle=h,
        pinned_tensor=pinned_memory)
    torch.save(f=dsfw, obj=buffer, _use_new_zipfile_serialization=True)
    write_sec = time.time() - st
    dsfw._dump_state()
    return write_sec

def run(mb_size, folder):
    buffer = torch.randint(high=128, size=(mb_size*(1024**2), ), dtype=torch.uint8, device='cpu').pin_memory()

    fn_dict = {
        'test_save': test_save, 
        'test_ds_mock_save': test_ds_mock_save, 
        'test_ds_py_save': test_ds_py_save,
        'test_ds_aio_save':test_ds_aio_save
    }
    for tag, fn in fn_dict.items():
        file = os.path.join(folder, f'{tag}_{mb_size}MB.pt')
        print(f'checkpoint file = {file}')
        if os.path.isfile(file):
            os.remove(file)
        st = time.time()
        write_sec = fn(file, buffer, True)
        gb_per_sec = mb_size/(1024.0*write_sec)
        gb_size = os.path.getsize(file)/(1024**3)
        print(f'{tag} -- {gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} gb/s')
        print(f'*********************************************')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        default=None,
                        type=str,
                        required=True,
                        help='Folder to use for I/O.')
    parser.add_argument('--mb_size',
                        type=int,
                        default=None,
                        required=True,
                        help='Size of tensor to save in MB.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args



def main():
    print(f'Performance test of deepspeed fast checkpoint')
    args = parse_arguments()
    if not os.path.exists(args.folder):
        print(f'Invalid folder: {args.folder}')
        quit()
    run(args.mb_size, args.folder)
    

if __name__ == "__main__":
    main()
