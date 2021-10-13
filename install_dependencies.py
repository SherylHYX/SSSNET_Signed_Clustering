"""
Installs dependencies in current conda environment.

Args:
    --gpu: Force installing a gpu version. When not specified, the script only installs a gpu version when cuda is detected.
"""

import sys
import subprocess
import argparse

from cuda_check import get_cuda_version

def is_torch_installed() -> bool:
    try:
        import torch
    except Exception:
        return False
    return True

def install_package(pkg_name:str):
    pkg_name = pkg_name.strip('\n')
    pkg_name_list = pkg_name.split(' ')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + pkg_name_list)

global_parser = argparse.ArgumentParser()
global_parser.add_argument("requirement_file", type=str, nargs='?', default="requirements.txt")
global_parser.add_argument("--gpu", dest='forcegpu', action='store_true')
# global_parser.add_argument("--no-gpu", dest='forcegpu', action='store_false')
global_parser.add_argument("--cpu", dest='forcecpu', action='store_true')
global_parser.set_defaults(forcecpu=False)
global_parser.set_defaults(forcegpu=False)

def cuda_canonical_version(v) ->str:
    if type(v) == int:
        # from the API (like 9020 for 9.2)
        major = v // 1000
        minor = (v - major * 1000) // 10
    elif type(v) == str:
        # from torch (like '10.2')
        major = int(v.split('.')[0])
        minor = int(v.split('.')[1])

    return 'cu' + str(major) + str(minor)

def main():
    global_args = global_parser.parse_args()

    if global_args.forcegpu and global_args.forcecpu:
        print("Conflicting args.")
        exit(-1)

    # is torch installed?
    b_torch_installed = is_torch_installed()
    cuda_version = 'cpu'
    torch_version = ''
    if b_torch_installed:
        print('Pytorch is installed. Determinating cuda version from pytorch...')
        # is cuda installed?
        import torch
        torch_version = torch.version.__version__
        if torch.version.cuda:
            cuda_version = cuda_canonical_version(torch.version.cuda)


        if cuda_version == 'cpu':
            print('Pytorch is CPU version')
        else:
            print(f'Pytorch Cuda version:{cuda_version}')
    else:
        # determine cuda version using our method
        try:
            cuda_ver_int = get_cuda_version()
            cuda_version = cuda_canonical_version(cuda_ver_int)
        except Exception as e:
            print('Get CUDA version failed. Fall back to cpu version')
            cuda_version = 'cpu'
        
        if global_args.forcegpu:
            # force installing our specified cuda version
            subprocess.check_call(['conda', 'install', 'cudatoolkit=10.2'])
            cuda_version = 'cu102'
            pass
        if global_args.forcecpu:
            cuda_version = 'cpu'

        # install torch
        torch_package_name = f'torch==1.8.1+{cuda_version} -f https://download.pytorch.org/whl/torch_stable.html'
        install_package(torch_package_name)


    # torch-scatter and torch-sparse
    TORCH_EXTENSION_URL = 'https://pytorch-geometric.com/whl/torch-1.8.0+{cuda_version}.html'
    torch_ext_dependencies = [
        'torch-sparse==0.6.9',
        'torch-scatter==2.0.6'
    ]
    torch_ext_dependencies = [' '.join([d, '-f', TORCH_EXTENSION_URL]) for d in torch_ext_dependencies]
    for d in torch_ext_dependencies:
        install_package(d)

    # other regular dependencies that can be easily installed via pypi
    with open(global_args.requirement_file) as f_req:
        dependencies = f_req.readlines()
    dependencies = [l for l in dependencies if not l.startswith('#')]
    for d in dependencies:
        install_package(d)
    pass

if __name__ == '__main__':
    main()