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

## 🐍 Python / NumPy Usage

The repository includes a thin Python interface backed by the native shared
library. It passes a contiguous NumPy buffer directly to the Odin C ABI, so a
reusable plan does not allocate or copy data during `forward` or `inverse`.

### Build the shared library

Build the Python-loadable library with the speed and CPU-specific options used
for native benchmarks:

```bash
make build ODIN=/path/to/odin ODIN_OPT=speed MICROARCH=native LIB_MODE=shared
```

This creates `build/libsleipnir_fft.so`. The Python module automatically looks
for that path relative to the repository. The module is not installed as a
wheel; add the repository `python/` directory to `PYTHONPATH`:

```bash
PYTHONPATH=python python3
```

The interface requires Python with NumPy. Matplotlib is additionally required
only by the benchmark-and-plot example.

### Reusable in-place plan

Import the public module as `sleipnirfft`:

```python
import numpy as np
from sleipnirfft import C2CPlan

n = 4096
rng = np.random.default_rng(7)
x = np.ascontiguousarray(
    rng.normal(size=n) + 1j * rng.normal(size=n),
    dtype=np.complex64,
)

with C2CPlan(n, x.dtype) as plan:
    spectrum = plan.forward(x)
    assert spectrum is x

    # The inverse is normalized by 1/N and also modifies x in-place.
    recovered = plan.inverse(spectrum)
    assert recovered is x
```

`C2CPlan` supports exactly `numpy.complex64` and `numpy.complex128`. The
transform input must be a one-dimensional, C-contiguous array whose length is
the plan length. Use `np.ascontiguousarray` for a sliced or otherwise
non-contiguous view:

```python
view = source[::2]
work = np.ascontiguousarray(view, dtype=np.complex128)
with C2CPlan(work.size, np.complex128) as plan:
    plan.forward(work)
```

The plan owns its native twiddle and scratch memory and should be closed with a
`with` block or `plan.close()`. Reusing a plan is preferred when many signals
have the same length, because plan creation is paid only once.

### One-shot functions

For convenience, `fft` and `ifft` make one contiguous output copy and leave the
input unchanged:

```python
import numpy as np
from sleipnirfft import fft, ifft

signal = np.ones(1024, dtype=np.complex128)
spectrum = fft(signal)
roundtrip = ifft(spectrum)
np.testing.assert_allclose(roundtrip, signal)
assert np.all(signal == 1.0 + 0.0j)
```

Use the reusable `C2CPlan` API for zero-copy in-place operation. Use one-shot
functions when simpler ownership behavior is more useful than avoiding the
input copy.

### NumPy comparison and spectrum plot

After building the shared library, run the included example from the project
root:

```bash
PYTHONPATH=python python3 python/benchmark_spectrum.py --dtype complex64 --n 65536 --warmup 5 --repeats 30 --output build/fft_spectrum_comparison.png
```

The script generates a deterministic multi-tone signal, measures the median
`np.fft.fft` time and the median reusable-plan Sleipnir time, reports the
maximum spectral difference, and overlays both spectra in the output PNG. Use
`--dtype complex128` for double precision, or `--show` to display the figure
as well as saving it. Plan creation time is reported separately and is not
included in repeated FFT timings.

A comparison detail is that standard NumPy FFT commonly promotes `complex64`
input to a `complex128` output. The example prints both output dtypes, so its
`complex64` comparison is explicitly a single-precision Sleipnir transform
versus NumPy output with promoted double precision.

### Custom shared-library path

When the library is outside the default `build/` directory, pass its path to
the plan or one-shot function:

```python
import numpy as np
from sleipnirfft import C2CPlan, fft

library = "/opt/sleipnir/lib/libsleipnir_fft.so"
data = np.zeros(2048, dtype=np.complex128)
with C2CPlan(data.size, data.dtype, library=library) as plan:
    plan.forward(data)
result = fft(data, library=library)
```
