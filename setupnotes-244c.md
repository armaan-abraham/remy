# Remy Setup Notes

## Dependencies
```bash
sudo apt-get install -y g++ protobuf-compiler libprotobuf-dev libboost-all-dev autoconf automake libtool
```

## Build Steps
```bash
./autogen.sh
./configure
make CXXFLAGS="-Wno-error=deprecated-copy" -j$(nproc)
ln -sf src/remy ./remy
```

## Key Issue
- Compilation fails with default flags due to deprecated-copy warnings
- Solution: Add `CXXFLAGS="-Wno-error=deprecated-copy"` to make command

## Verify
```bash
./remy  # Should ask for config file
```

## LibTorch
LibTorch installed in `libtorch/`. Test: `make -C tests libtorch-test && tests/libtorch-test`