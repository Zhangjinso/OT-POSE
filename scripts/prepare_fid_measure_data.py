import os
import lmdb
import argparse
import multiprocessing
from PIL import Image
from tqdm import tqdm
from io import BytesIO

from torchvision.transforms import functional as trans_fn


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(5)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class Resizer:
    def __init__(self, *, size, root):
        self.size = size
        self.root = root

    def get_resized_bytes(self, img):
        img = trans_fn.resize(img, self.size, Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format='png')
        img_bytes = buf.getvalue()
        return img_bytes

    def prepare(self, filename):
        filename = os.path.join(self.root, filename)
        img = Image.open(filename)
        img = img.convert('RGB')
        img = img.resize((self.size[1],self.size[0]), Image.BICUBIC)
        # img_bytes = self.get_resized_bytes(img)
        return img

    def __call__(self, filename):
        result_img = self.prepare(filename)
        return result_img, filename


def prepare_data(root, dataset, out, n_worker, sizes, chunksize):
    assert dataset in ['deepfashion']
    if dataset == 'deepfashion':
        file_txt = '{}/test_pairs.txt'.format(root)
        filenames = []
        with open(file_txt, 'r') as f:
            lines = f.readlines()
            for item in lines:
                filenames.extend(item.strip().split(','))

        # file_txt = '{}/test_pairs.txt'.format(root)
        # with open(file_txt, 'r') as f:
        #     lines = f.readlines()
        #     for item in lines:
        #         filenames.extend(item.strip().split(','))                
        # filenames = list(set(filenames))


    total = len(filenames)
    os.makedirs(out, exist_ok=True)

    for size in sizes:
        out1 = os.path.join(out, str(size[0]))
        os.makedirs(out1, exist_ok=True)
        resizer = Resizer(size=size, root=root)
        for filename in tqdm(filenames):
            result_img, filename = resizer(filename)
            filename = os.path.splitext(filename)[0] + '.png'
            result_img.save(os.path.join(out1,'_'.join(filename.split('/'))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='a path to output directory')
    parser.add_argument('--dataset', type=str, default='deepfashion', help='a path to output directory')
    parser.add_argument('--out', type=str, help='a path to output directory')
    parser.add_argument('--sizes', type=int, nargs='+', default=((512, 352),(256, 176)) )
    parser.add_argument('--n_worker', type=int, help='number of worker processes', default=8)
    parser.add_argument('--chunksize', type=int, help='approximate chunksize for each worker', default=10)
    args = parser.parse_args()
    prepare_data(**vars(args))