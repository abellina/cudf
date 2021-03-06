# Contributing to cuDF

Contributions to cuDF fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/rapidsai/cudf/issues/new/choose)
    describing in detail the problem or new feature. The RAPIDS team evaluates 
    and triages issues, and schedules them for a release. If you believe the 
    issue needs priority attention, please comment on the issue to notify the 
    team.
2. To propose and implement a new Feature, please file a new feature request 
    [issue](https://github.com/rapidsai/cudf/issues/new/choose). Describe the 
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and 
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please 
    Follow the [code contributions](#code-contributions) guide below. If you 
    need more context on a particular issue, please ask in a comment.

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for [Setting Up Your Build Environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.
3. Comment on the issue stating that you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/cudf/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a RAPIDS developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues for our next release in our [project boards](https://github.com/rapidsai/cudf/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable
contributing. Start with _Step 3_ above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

The following instructions are for developers and contributors to cuDF OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuDF from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

### Get libcudf Dependencies

Compiler requirements:

* `gcc`     version 5.4+
* `nvcc`    version 9.2+
* `cmake`   version 3.12.4+

CUDA/GPU requirements:

* CUDA 9.2+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

Since `cmake` will download and build Apache Arrow you may need to install Boost C++ (version 1.58+) before running
`cmake`:

```bash
# Install Boost C++ for Ubuntu 16.04/18.04
$ sudo apt-get install libboost-all-dev
```

or

```bash
# Install Boost C++ for Conda
$ conda install -c conda-forge boost
```

## Script to build cuDF from source

### Build from Source

To install cuDF from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules
```bash
CUDF_HOME=$(pwd)/cudf
git clone https://github.com/rapidsai/cudf.git $CUDF_HOME
cd $CUDF_HOME
git submodule update --init --remote --recursive
```
- Create the conda development environment `cudf_dev`:
```bash
# create the conda environment (assuming in base `cudf` directory)
conda env create --name cudf_dev --file conda/environments/cudf_dev_cuda10.0.yml
# activate the environment
source activate cudf_dev
```
- If you're using CUDA 9.2, you will need to create the environment with `conda env create --name cudf_dev --file conda/environments/cudf_dev_cuda9.2.yml` instead.

- Build and install `libcudf`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
```bash
$ cd $CUDF_HOME/cpp                                                       # navigate to C/C++ CUDA source root directory
$ mkdir build                                                             # make a build directory
$ cd build                                                                # enter the build directory

# CMake options:
# -DCMAKE_INSTALL_PREFIX set to the install path for your libraries or $CONDA_PREFIX if you're using Anaconda, i.e. -DCMAKE_INSTALL_PREFIX=/install/path or -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# -DCMAKE_CXX11_ABI set to ON or OFF depending on the ABI version you want, defaults to ON. When turned ON, ABI compability for C++11 is used. When OFF, pre-C++11 ABI compability is used.
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON      # configure cmake ...

$ make -j                                                                 # compile the libraries librmm.so, libcudf.so ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                                            # install the libraries librmm.so, libcudf.so to the CMAKE_INSTALL_PREFIX
```

- As a convenience, a `build.sh` script is provided in `$CUDF_HOME`. To execute the same build commands above, run the script as shown below.  Note that the libraries will be installed to the location set in `$INSTALL_PREFIX` if set (i.e. `export INSTALL_PREFIX=/install/path`), otherwise to `$CONDA_PREFIX`.
```bash
$ cd $CUDF_HOME
$ ./build.sh libcudf                   # compile the cuDF libraries and install them to $INSTALL_PREFIX if set, otherwise $CONDA_PREFIX
```

- To run tests (Optional):
```bash
$ make test
```

- Build the `cudf` python package, in the `python` folder:
```bash
$ cd $CUDF_HOME/python
$ python setup.py build_ext --inplace
```

- Like the `libcudf` build step above, `build.sh` can also be used to build the `cudf` python package, as shown below:
```bash
$ cd $CUDF_HOME
$ ./build.sh cudf
```

- You will also need the following environment variables, including `$CUDA_HOME`.
```bash
NUMBAPRO_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
NUMBAPRO_LIBDEVICE=$CUDA_HOME/nvvm/libdevice
```

- To run Python tests (Optional):
```bash
$ py.test -v                           # run python tests on cudf python bindings
```

- Other `build.sh` options:
```bash
$ cd $CUDF_HOME
$ ./build.sh clean                     # remove any prior build artifacts and configuration (start over)
$ ./build.sh libcudf -v                # compile and install libcudf with verbose output
$ ./build.sh libcudf -g                # compile and install libcudf for debug
$ PARALLEL_LEVEL=4 ./build.sh libcudf  # compile and install libcudf limiting parallel build jobs to 4 (make -j4)
$ ./build.sh libcudf -n                # compile libcudf but do not install
```

- The `build.sh` script can be customized to support other features:
  - **ABI version:** The cmake -DCMAKE_CXX11_ABI option can be set to ON or OFF depending on the ABI version you want, defaults to ON. When turned ON, ABI compability for C++11 is used. When OFF, pre-C++11 ABI compability is used.


Done! You are ready to develop for the cuDF OSS project.

## Debugging cuDF

### Building Debug mode from source

Follow the [above instructions](#build-from-source) to build from source and add `-DCMAKE_BUILD_TYPE=Debug` to the `cmake` step. 

For example:
```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path -DCMAKE_BUILD_TYPE=Debug     # configure cmake ... use -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX if you're using Anaconda
```

This builds `libcudf` in Debug mode which enables some `assert` safety checks and includes symbols in the library for debugging.

All other steps for installing `libcudf` into your environment are the same.

### Debugging with `cuda-gdb` and `cuda-memcheck`

When you have a debug build of `libcudf` installed, debugging with the `cuda-gdb` and `cuda-memcheck` is easy.

If you are debugging a Python script, simply run the following:

#### `cuda-gdb`

```bash
cuda-gdb -ex r --args python <program_name>.py <program_arguments>
```

#### `cuda-memcheck`

```bash
cuda-memcheck python <program_name>.py <program_arguments>
```

### Building and Testing on a gpuCI image locally

Before submitting a pull request, you can do a local build and test on your machine that mimics our gpuCI environment using the `ci/local/build.sh` script.
For detailed information on usage of this script, see [here](ci/local/README.md).

## Automated Build in Docker Container

A Dockerfile is provided with a preconfigured conda environment for building and installing cuDF from source based off of the master branch.

### Prerequisites

* Install [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) for Docker + GPU support
* Verify NVIDIA driver is `396.44` or higher
* Ensure CUDA 9.2+ is installed

### Usage

From cudf project root run the following, to build with defaults:
```bash
$ docker build --tag cudf .
```
After the container is built run the container:
```bash
$ docker run --runtime=nvidia -it cudf bash
```
Activate the conda environment `cudf` to use the newly built cuDF and libcudf libraries:
```
root@3f689ba9c842:/# source activate cudf
(cudf) root@3f689ba9c842:/# python -c "import cudf"
(cudf) root@3f689ba9c842:/#
```

### Customizing the Build

Several build arguments are available to customize the build process of the
container. These are specified by using the Docker [build-arg](https://docs.docker.com/engine/reference/commandline/build/#set-build-time-variables---build-arg)
flag. Below is a list of the available arguments and their purpose:

| Build Argument | Default Value | Other Value(s) | Purpose |
| --- | --- | --- | --- |
| `CUDA_VERSION` | 9.2 | 10.0 | set CUDA version |
| `LINUX_VERSION` | ubuntu16.04 | ubuntu18.04 | set Ubuntu version |
| `CC` & `CXX` | 5 | 7 | set gcc/g++ version; **NOTE:** gcc7 requires Ubuntu 18.04 |
| `CUDF_REPO` | This repo | Forks of cuDF | set git URL to use for `git clone` |
| `CUDF_BRANCH` | master | Any branch name | set git branch to checkout of `CUDF_REPO` |
| `NUMBA_VERSION` | newest | >=0.40.0 | set numba version |
| `NUMPY_VERSION` | newest | >=1.14.3 | set numpy version |
| `PANDAS_VERSION` | newest | >=0.23.4 | set pandas version |
| `PYARROW_VERSION` | 0.12.1 | Not supported | set pyarrow version |
| `CMAKE_VERSION` | newest | >=3.12 | set cmake version |
| `CYTHON_VERSION` | 0.29 | Not supported | set Cython version |
| `PYTHON_VERSION` | 3.6 | 3.7 | set python version |

---

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
