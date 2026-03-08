# sleipnir-fft

Fast FFT library package for Odin.

Core package path in this repository:

- `src` (import as `...:src`)

## Use As Dependency

Example project layout:

```text
your_project/
  third_party/
    sleipnir-fft/      # clone this repository here
  src/
    main.odin
```

Build:

```bash
odin run src -collection:sleipnirfft=third_party/sleipnir-fft
```

Import from code:

```odin
import fft "sleipnirfft:src"
```

## FFTW-style Build/Install

This repository also supports `configure` + `make` workflow:

```bash
./configure --prefix=$HOME/.local --mode=static --microarch=native
make -j
make install
```

`configure` auto-detects `odin` from `PATH` (equivalent to `which odin`).
You can also override it with `--odin=/path/to/odin` (or `ODIN=/path/to/odin ./configure`).

After install, point Odin collection alias to the installed package root:

```bash
odin run src -collection:sleipnirfft=$HOME/.local/share/sleipnir-fft
```

Main targets:

- `make build` : build library artifact (`build/libsleipnir_fft.a` etc.)
- `make test` : run `src` package tests
- `make install` : install source package + built library
- `make install-src` : install source package only
- `make install-lib` : install built library only
- `make uninstall` : remove installed package/library from `PREFIX`
- `make print-config` : show effective configuration

## Quick Smoke Test

From this repository root:

```bash
./scripts/run_package_consumer_example.sh
```

## API/Details

See:

- `src/README.md`
