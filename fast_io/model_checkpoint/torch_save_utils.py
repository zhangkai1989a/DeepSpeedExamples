import time
import torch
import os
import deepspeed
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.io import MockFileWriter, PyFileWriter, FastFileWriter


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

def test_save(file, buffer, use_zipfile, io_buffer_mb):
    st = time.time()
    torch.save(f=file, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    return time.time() - st


def test_ds_mock_save(file, buffer, use_zipfile, io_buffer_mb):
    st = time.time()
    ds_mock_writer = MockFileWriter(file)
    torch.save(f=ds_mock_writer, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    ds_mock_writer.close() # Force flush to storage    
    write_sec = time.time() - st
    ds_mock_writer._dump_state()
    return write_sec 

def test_ds_py_save(file, buffer, use_zipfile, io_buffer_mb):
    st = time.time()
    ds_py_writer = PyFileWriter(file)   
    torch.save(f=ds_py_writer, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    ds_py_writer.close() # Force flush to storage
    write_sec = time.time() - st
    ds_py_writer._dump_state()
    return write_sec 

def test_ds_fast_save(file, buffer, use_zipfile, io_buffer_mb):
    h = _get_aio_handle()
    pinned_memory = torch.zeros(io_buffer_mb*(1024**2), dtype=torch.uint8, device='cpu').pin_memory()                                            
    st = time.time()
    ds_fast_writer = FastFileWriter(
        file_path=file, 
        aio_handle=h,
        pinned_tensor=pinned_memory)
    torch.save(f=ds_fast_writer, obj=buffer, _use_new_zipfile_serialization=use_zipfile)
    ds_fast_writer.close() # Force flush to storage
    write_sec = time.time() - st
    ds_fast_writer._dump_state()
    return write_sec