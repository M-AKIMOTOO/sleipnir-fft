# sleipnir-fft/src (Odin)

Low-allocation radix-2 FFT package for Odin.

## Features

- Complex-to-complex FFT/IFFT (`complex128`, `complex64`)
- 2D complex-to-complex FFT/IFFT (`rows x cols`, row-major)
- Real-to-complex FFT and inverse (`f64 <-> complex128`)
- Reusable plans with precomputed twiddles
- No allocations during execution (`*_forward`, `*_inverse`)
- Optional external scratch APIs for tighter memory control
- Backend selection (`Auto`, `Cooley_Tukey`, `Split_Radix`)

## Import

Recommended dependency layout:

```text
your_project/
  third_party/
    sleipnir-fft/         # this repository
  src/
    main.odin
```

Build with collection alias pointing to repository root:

```bash
odin run src -collection:sleipnirfft=third_party/sleipnir-fft
```

If installed via `./configure && make install` (default prefix `/usr/local`), use:

```bash
odin run src -collection:sleipnirfft=/usr/local/share/sleipnir-fft
```

Then import:

```odin
import fft "sleipnirfft:src"
```

## Example

```odin
package main

import "core:fmt"
import fft "sleipnirfft:src"

main :: proc() {
	plan: fft.C2C_Plan
	// Default c2c_plan_init uses Auto backend selection.
	if fft.c2c_plan_init(&plan, 8) != .None {
		return
	}
	defer fft.c2c_plan_destroy(&plan)

	x := []complex128{
		1, 0, 0, 0, 0, 0, 0, 0,
	}
	_ = fft.c2c_forward_in_place(&plan, x)
	fmt.println(x)
}
```

## Notes

- C2C FFT supports arbitrary length. Power-of-two uses Cooley/Split kernels; non-power-of-two uses Bluestein fallback.
- R2C/C2R supports arbitrary `N`. Power-of-two uses packed-half optimization; non-power-of-two falls back to full C2C path.
- 2D C2C supports arbitrary `rows x cols`; each axis uses the same backend selection/fallback as 1D C2C.
- If `Split_Radix` is requested for non-power-of-two length, it is automatically downgraded to Cooley/Bluestein.
- `c2c_inverse_in_place` and `c2r_inverse` are normalized (divide by `N`).
- For low-memory mode, use `*_with_scratch` to supply your own `[]complex128` scratch.
- Algorithm files are split by backend (`cooley_tukey.odin`, `split_radix.odin`).
- `Auto` uses a deterministic base heuristic, then (for power-of-two `1024..262144`) runs a short runtime autotune to choose backend/radix/thread config on the current machine.
- Autotune selection uses a deadband/tie-break rule to reduce noise-driven flips between very close candidates.
- Split-radix is included as an autotune candidate for smaller power-of-two sizes (`<=8192`).
- `Auto` in Cooley-Tukey path picks radix (`2`/`4`) by size and shape (`cooley_radix=0`).
- `complex64` API is available via `*_f32` procedures (`c2c_plan_init_f32`, `c2c_forward_in_place_f32`, ...).
- `*_f32` supports arbitrary `N`; non-power-of-two uses an internal `f64` fallback plan (higher compatibility, slower than pure f32 kernel).
- Low-RAM initialization is available via `c2c_plan_init_low_ram` / `c2c_plan_init_f32_low_ram` (skips inverse twiddle table and bit-reversal table to reduce plan memory, with slower execution).
- Threaded Cooley-Tukey mode can be enabled by `C2C_Plan_Options.threads` (`0` auto, >1 explicit), for both radix-2 and radix-4 kernels. In auto-thread mode, this implementation currently uses up to 4 threads for larger sizes.
- Radix-2 AVX2 path packs stage twiddles at plan initialization (forward/inverse) to reduce per-stage twiddle setup overhead.
- Radix-4 kernels include an AVX2 fast path (`amd64 + avx2` target feature, `core:simd`, 2 butterflies per inner iteration), with scalar fallback.
- For radix-4 AVX2 path, twiddles are prepacked at plan init for vector loads (faster execution, higher plan memory).
- Cooley radix can be selected by `C2C_Plan_Options.cooley_radix` (`0=auto`, `2`, `4`).
- Plan memory can be estimated with `c2c_plan_estimate_bytes` / `c2c_plan_estimate_bytes_f32`.
- Package version constants are exposed as `fft.VERSION_MAJOR`, `fft.VERSION_MINOR`, `fft.VERSION_PATCH`, `fft.VERSION_STRING`.

## Benchmark

From workspace root:

```bash
odin run sleipnir-fft/fft_bench.odin -file -collection:sleipnirfft=./sleipnir-fft
```

You can compare builds like:

```bash
odin run sleipnir-fft/fft_bench.odin -file -collection:sleipnirfft=./sleipnir-fft -o:speed
odin run sleipnir-fft/fft_bench.odin -file -collection:sleipnirfft=./sleipnir-fft -o:speed -microarch:native
```

Or run automatic comparison:

```bash
./scripts/bench_fft_compare.sh
```
