# Remy Setup Notes

## Dependencies

These are dependencies for the original remy project.
```bash
sudo apt-get install -y g++ protobuf-compiler libprotobuf-dev libboost-all-dev autoconf automake libtool
```

## Build Steps

```bash
autoreconf -fi # optional
./autogen.sh
./configure # may need to prepend stuff, see libtorch note
make clean # optional
make CXXFLAGS="-Wno-error=deprecated-copy" -j$(nproc)
```

## LibTorch

LibTorch installed in `libtorch/`. Test: `make -C tests libtorch-test && tests/libtorch-test`

LibTorch 2.10.0 bundles protobuf 3.13.0 headers. The project build links against the system protobuf library via pkg-config, so the system version must match LibTorch's bundled version.

One way around this is a conda env with `libprotobuf=3.13.0`. You must also set `PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH` when running `./configure`, since conda activation alone doesn't always prepend it. e.g.:
```bash
PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH ./configure
```

Another brute force way around this is to rebuild torchlib from source.

## Run

```bash
# Must provide configs to both via cf=:
./src/remy       # original
./src/rattrainer # RL
```