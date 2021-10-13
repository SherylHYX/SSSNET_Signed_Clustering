#/bin/bash
env_name=$1
if [ -z "$env_name" ]
then
    echo "Using default env name (SSSNET)"
    env_name="SSSNET"
fi
conda create -y -n $env_name python=3.6
conda activate $env_name

# cuda
conda install -n $env_name -c anaconda cudatoolkit=10.2
# other python dependencies
python install_dependencies.py