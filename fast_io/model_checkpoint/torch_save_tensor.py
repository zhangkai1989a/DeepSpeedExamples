import time
import argparse
import torch
import os
from torch_save_utils import PINNED_BUFFER_MB
from torch_save_utils import test_save, test_ds_mock_save, test_ds_py_save, test_ds_fast_save


def run(mb_size, folder, legacy_save, io_buffer_mb, device):
    buffer = torch.randint(high=128,
                           size=(mb_size * (1024**2), ),
                           dtype=torch.uint8,
                           device=device)

    fn_dict = {
        'test_save': test_save,
        'test_ds_mock_save': test_ds_mock_save,
        'test_ds_py_save': test_ds_py_save,
        'test_ds_fast_save': test_ds_fast_save
    }
    for tag, fn in fn_dict.items():
        file = os.path.join(folder, f'{tag}_{mb_size}MB.pt')
        print(f'checkpoint file = {file}')
        if os.path.isfile(file):
            os.remove(file)
        st = time.time()
        write_sec = fn(file, buffer, not legacy_save, io_buffer_mb)
        gb_per_sec = mb_size / (1024.0 * write_sec)
        gb_size = os.path.getsize(file) / (1024**3)
        print(
            f'{tag} -- {gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} gb/s'
        )
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
    parser.add_argument('--legacy',
                        action='store_true',
                        help='Use torch legacy save format')

    parser.add_argument('--gpu', action='store_true', help='Use gpu tensors.')

    parser.add_argument('--io_buffer_mb',
                        type=int,
                        default=PINNED_BUFFER_MB,
                        help='Size of pinned i/o buffer in MB.')

    args = parser.parse_args()
    print(f'args = {args}')
    return args


def main():
    print(f'Performance test of deepspeed fast tensor checkpoint')
    args = parse_arguments()
    if not os.path.exists(args.folder):
        print(f'Invalid folder: {args.folder}')
        quit()

    device = torch.cuda.current_device() if args.gpu else 'cpu'
    run(args.mb_size, args.folder, args.legacy, args.io_buffer_mb, device)


if __name__ == "__main__":
    main()
