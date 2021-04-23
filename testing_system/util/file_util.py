import os
import shutil
from contextlib import contextmanager

from detect_delimiter import detect


def create_py_file(src, dist_dir, fname):
    path = os.path.join(dist_dir, fname)
    with open(path, 'w') as out_f:
        out_f.write(src)


def get_file_name_from_full_path(path_to_file: str):
    return path_to_file.split('/')[-1]


def get_dir_name_from_full_path(path_to_dir: str):
    return path_to_dir.split('/')[-1]


def copytree_to_existing_dir(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def find_all_files_with_ext(path_to_dir, ext, full_paths=True):
    res = []
    for root, dirs, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith(f'.{ext}'):
                if full_paths:
                    res.append(os.path.join(root, file))
                else:
                    res.append(file)
    return res


def detect_csv_delimiter(path_to_csv):
    with open(path_to_csv) as f:
        first_line = f.readline()
    delimiter = detect(first_line)
    return delimiter


@contextmanager
def cwd(path):
    old_pwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_pwd)
