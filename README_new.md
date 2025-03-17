Learn SuperPoint network and try it in Python and C++.

# Python runner

```zsh
# Follow original README to setup an Python env

# Activate environment
. ~/.venv/SuperPoint/bin/Activate

# Run feature extraction 
python ./feature_extraction.py /Users/bowen/Source/dataset/rgbd_dataset_freiburg1_xyz/rgb [--waitkey 1000000]
```

# Convert to libTorch model for C++

```zsh
# Activate environment
. ~/.venv/SuperPoint/bin/Activate

# Convert model
python3 convert_model_for_cpp.py
```

# Use model in C++

1. Download libtorch binary from [here](https://download.pytorch.org/libtorch/cpu/).
    * For Intel-mac, need to use `libtorch-macos-x86_64-2.2.2.zip`
2. Unzip the file into a folder
3. Build cpp runner
```zsh
# Build it
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/Users/bowen/Source/3rd_party/libtorch;/Users/bowen/Source/3rd_party/build_opencv" ..
cmake --build . --config Release

# Run the tester
./superpoint_runner ../superpoint_converted.pt /Users/bowen/Source/dataset/rgbd_dataset_freiburg1_xyz
```
