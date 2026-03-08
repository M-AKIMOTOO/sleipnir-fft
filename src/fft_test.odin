package fft

import "core:math"
import "core:math/rand"
import "core:testing"

EPS :: 1e-9

abs_f64 :: #force_inline proc(x: f64) -> f64 {
	if x < 0 {
		return -x
	}
	return x
}

approx_eq_f64 :: #force_inline proc(a, b, eps: f64) -> bool {
	return abs_f64(a-b) <= eps
}

approx_eq_c128 :: #force_inline proc(a, b: complex128, eps: f64) -> bool {
	return approx_eq_f64(real(a), real(b), eps) && approx_eq_f64(imag(a), imag(b), eps)
}

approx_eq_c64 :: #force_inline proc(a, b: complex64, eps: f64) -> bool {
	return approx_eq_f64(f64(real(a)), f64(real(b)), eps) && approx_eq_f64(f64(imag(a)), f64(imag(b)), eps)
}

naive_dft_forward :: proc(input: []complex128, output: []complex128) {
	n := len(input)
	inv_n := 1.0 / f64(n)
	for k in 0..<n {
		sum := complex(0.0, 0.0)
		for t in 0..<n {
			angle := -2.0 * math.PI * f64(k*t) * inv_n
			s, c := math.sincos(angle)
			sum += input[t] * complex(c, s)
		}
		output[k] = sum
	}
}

@test
test_c2c_roundtrip :: proc(t: ^testing.T) {
	n := 16
	plan: C2C_Plan
	err := c2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "c2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)

	data, data_alloc_err := make([]complex128, n)
	orig, orig_alloc_err := make([]complex128, n)
	if data_alloc_err != .None || orig_alloc_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_alloc_err, orig_alloc_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		v := f64(i)
		data[i] = complex(math.sin(0.33*v)+0.1*v, math.cos(0.27*v)-0.05*v)
	}
	copy(orig, data)

	err = c2c_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_forward_in_place failed: %v", err)
	if err != .None {
		return
	}

	err = c2c_inverse_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_inverse_in_place failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], EPS), "roundtrip mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_roundtrip_non_power_of_two :: proc(t: ^testing.T) {
	n := 1000
	plan: C2C_Plan
	err := c2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "c2c_plan_init(non-power-of-two) failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)
	testing.expectf(t, plan.uses_bluestein, "expected bluestein path for non-power-of-two length")

	data, data_err := make([]complex128, n)
	orig, orig_err := make([]complex128, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		x := f64(i)
		data[i] = complex(math.sin(0.31*x)+0.07*x, math.cos(0.19*x)-0.03*x)
	}
	copy(orig, data)

	err = c2c_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_forward_in_place failed: %v", err)
	if err != .None {
		return
	}
	err = c2c_inverse_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_inverse_in_place failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], 3e-8), "non-power-of-two roundtrip mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_non_power_of_two_matches_naive_dft :: proc(t: ^testing.T) {
	n := 15
	plan: C2C_Plan
	err := c2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "c2c_plan_init(non-power-of-two) failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)

	a, a_err := make([]complex128, n)
	ref, ref_err := make([]complex128, n)
	if a_err != .None || ref_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", a_err, ref_err)
		return
	}
	defer delete(a)
	defer delete(ref)

	for i in 0..<n {
		x := f64(i)
		a[i] = complex(math.sin(0.37*x), math.cos(0.23*x))
	}
	naive_dft_forward(a, ref)
	err = c2c_forward_in_place(&plan, a)
	testing.expectf(t, err == .None, "c2c_forward_in_place failed: %v", err)
	if err != .None {
		return
	}
	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(a[i], ref[i], 1e-10), "naive mismatch at %d: got=%v ref=%v", i, a[i], ref[i])
	}
}

@test
test_c2c_roundtrip_f32 :: proc(t: ^testing.T) {
	n := 64
	plan: C2C_Plan_F32
	err := c2c_plan_init_f32(&plan, n)
	testing.expectf(t, err == .None, "c2c_plan_init_f32 failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy_f32(&plan)

	data, data_err := make([]complex64, n)
	orig, orig_err := make([]complex64, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		x := f64(i)
		data[i] = complex(
			f32(math.sin(0.33*x) + 0.1*x),
			f32(math.cos(0.27*x) - 0.05*x),
		)
	}
	copy(orig, data)

	err = c2c_forward_in_place_f32(&plan, data)
	testing.expectf(t, err == .None, "c2c_forward_in_place_f32 failed: %v", err)
	if err != .None {
		return
	}

	err = c2c_inverse_in_place_f32(&plan, data)
	testing.expectf(t, err == .None, "c2c_inverse_in_place_f32 failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c64(data[i], orig[i], 4e-5), "f32 roundtrip mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_roundtrip_f32_non_power_of_two :: proc(t: ^testing.T) {
	n := 1000
	plan: C2C_Plan_F32
	err := c2c_plan_init_f32(&plan, n)
	testing.expectf(t, err == .None, "c2c_plan_init_f32 failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy_f32(&plan)
	testing.expectf(t, plan.uses_f64_fallback, "expected f64 fallback for non-power-of-two f32 plan")

	data, data_err := make([]complex64, n)
	orig, orig_err := make([]complex64, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		x := f64(i)
		data[i] = complex(
			f32(math.sin(0.11*x) + 0.05*math.cos(0.07*x)),
			f32(math.cos(0.09*x) - 0.04*math.sin(0.05*x)),
		)
	}
	copy(orig, data)

	err = c2c_forward_in_place_f32(&plan, data)
	testing.expectf(t, err == .None, "c2c_forward_in_place_f32 failed: %v", err)
	if err != .None {
		return
	}

	err = c2c_inverse_in_place_f32(&plan, data)
	testing.expectf(t, err == .None, "c2c_inverse_in_place_f32 failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c64(data[i], orig[i], 8e-4), "f32 non-power-of-two mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_roundtrip_low_ram :: proc(t: ^testing.T) {
	n := 128
	plan: C2C_Plan
	err := c2c_plan_init_low_ram(&plan, n, .Cooley_Tukey)
	testing.expectf(t, err == .None, "c2c_plan_init_low_ram failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)
	testing.expectf(t, plan.twiddles_inv == nil, "low_ram plan should not allocate twiddles_inv")
	testing.expectf(t, plan.bitrev == nil, "low_ram plan should not allocate bitrev table")

	data, data_err := make([]complex128, n)
	orig, orig_err := make([]complex128, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		x := f64(i)
		data[i] = complex(math.sin(0.21*x), math.cos(0.13*x))
	}
	copy(orig, data)
	_ = c2c_forward_in_place(&plan, data)
	_ = c2c_inverse_in_place(&plan, data)

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], 2e-9), "low_ram mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_plan_estimate_bytes_low_ram :: proc(t: ^testing.T) {
	n := 1 << 20
	normal := c2c_plan_estimate_bytes(n, .Cooley_Tukey, true, true)
	low_ram := c2c_plan_estimate_bytes(n, .Cooley_Tukey, false, false)
	testing.expectf(t, normal > 0, "normal estimate should be positive")
	testing.expectf(t, low_ram > 0, "low_ram estimate should be positive")
	testing.expectf(t, low_ram < normal, "low_ram estimate should be smaller: low_ram=%d normal=%d", low_ram, normal)
}

@test
test_c2c_plan_estimate_bytes_f32_low_ram :: proc(t: ^testing.T) {
	n := 1 << 20
	normal := c2c_plan_estimate_bytes_f32(n, .Cooley_Tukey, true, true)
	low_ram := c2c_plan_estimate_bytes_f32(n, .Cooley_Tukey, false, false)
	testing.expectf(t, normal > 0, "normal f32 estimate should be positive")
	testing.expectf(t, low_ram > 0, "low_ram f32 estimate should be positive")
	testing.expectf(t, low_ram < normal, "low_ram f32 estimate should be smaller: low_ram=%d normal=%d", low_ram, normal)
}

@test
test_c2c_plan_threads_option :: proc(t: ^testing.T) {
	n := 1024
	plan: C2C_Plan
	opts := C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 4,
		cooley_radix = 2,
	}
	err := c2c_plan_init_with_options(&plan, n, opts)
	testing.expectf(t, err == .None, "c2c_plan_init_with_options failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)

	testing.expectf(t, plan.num_threads == 4, "unexpected thread count: got=%d want=4", plan.num_threads)
}

@test
test_c2c_radix4_matches_radix2 :: proc(t: ^testing.T) {
	n := 256
	p2: C2C_Plan
	p4: C2C_Plan

	err := c2c_plan_init_with_options(&p2, n, C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 0,
		cooley_radix = 2,
	})
	testing.expectf(t, err == .None, "radix2 plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&p2)

	err = c2c_plan_init_with_options(&p4, n, C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 0,
		cooley_radix = 4,
	})
	testing.expectf(t, err == .None, "radix4 plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&p4)

	a, a_err := make([]complex128, n)
	b, b_err := make([]complex128, n)
	if a_err != .None || b_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", a_err, b_err)
		return
	}
	defer delete(a)
	defer delete(b)

	for i in 0..<n {
		x := f64(i)
		v := complex(math.sin(0.17*x)+0.2*math.cos(0.09*x), math.cos(0.21*x)-0.1*math.sin(0.13*x))
		a[i] = v
		b[i] = v
	}

	err = c2c_forward_in_place(&p2, a)
	testing.expectf(t, err == .None, "radix2 forward failed: %v", err)
	if err != .None {
		return
	}
	err = c2c_forward_in_place(&p4, b)
	testing.expectf(t, err == .None, "radix4 forward failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(a[i], b[i], 2e-9), "radix2/radix4 mismatch at %d: r2=%v r4=%v", i, a[i], b[i])
	}
}

@test
test_c2c_auto_radix_selection :: proc(t: ^testing.T) {
	p_small: C2C_Plan
	err := c2c_plan_init_with_options(&p_small, 1024, C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 0,
		cooley_radix = 0,
	})
	testing.expectf(t, err == .None, "small plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&p_small)
	testing.expectf(t, p_small.cooley_radix == 4, "expected radix-4 for 1024, got %d", p_small.cooley_radix)

	p_large: C2C_Plan
	err = c2c_plan_init_with_options(&p_large, 4096, C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 0,
		cooley_radix = 0,
	})
	testing.expectf(t, err == .None, "large plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&p_large)
	testing.expectf(t, p_large.cooley_radix == 4, "expected radix-4 for 4096, got %d", p_large.cooley_radix)
}

@test
test_c2c_radix4_parallel_roundtrip :: proc(t: ^testing.T) {
	n := 4096
	plan: C2C_Plan
	err := c2c_plan_init_with_options(&plan, n, C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 4,
		cooley_radix = 4,
	})
	testing.expectf(t, err == .None, "radix4 parallel plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)

	data, data_err := make([]complex128, n)
	orig, orig_err := make([]complex128, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		x := f64(i)
		data[i] = complex(math.sin(0.23*x)+0.1*math.cos(0.07*x), math.cos(0.17*x)-0.05*math.sin(0.11*x))
	}
	copy(orig, data)

	err = c2c_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "radix4 parallel forward failed: %v", err)
	if err != .None {
		return
	}
	err = c2c_inverse_in_place(&plan, data)
	testing.expectf(t, err == .None, "radix4 parallel inverse failed: %v", err)
	if err != .None {
		return
	}
	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], 3e-9), "radix4 parallel mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_f32_matches_f64_roughly :: proc(t: ^testing.T) {
	n := 128
	plan64: C2C_Plan
	plan32: C2C_Plan_F32

	err := c2c_plan_init_with_backend(&plan64, n, .Cooley_Tukey)
	testing.expectf(t, err == .None, "c2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan64)

	err = c2c_plan_init_f32_with_backend(&plan32, n, .Cooley_Tukey)
	testing.expectf(t, err == .None, "c2c_plan_init_f32_with_backend failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy_f32(&plan32)

	a64, a64_err := make([]complex128, n)
	a32, a32_err := make([]complex64, n)
	if a64_err != .None || a32_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", a64_err, a32_err)
		return
	}
	defer delete(a64)
	defer delete(a32)

	for i in 0..<n {
		x := f64(i)
		re := math.sin(2.0*math.PI*x/17.0) + 0.2*math.cos(2.0*math.PI*x/9.0)
		im := 0.4*math.sin(2.0*math.PI*x/13.0) - 0.3*math.cos(2.0*math.PI*x/11.0)
		a64[i] = complex(re, im)
		a32[i] = complex(f32(re), f32(im))
	}

	err = c2c_forward_in_place(&plan64, a64)
	testing.expectf(t, err == .None, "c2c_forward_in_place failed: %v", err)
	if err != .None {
		return
	}
	err = c2c_forward_in_place_f32(&plan32, a32)
	testing.expectf(t, err == .None, "c2c_forward_in_place_f32 failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(
			t,
			approx_eq_f64(f64(real(a32[i])), real(a64[i]), 1e-3) && approx_eq_f64(f64(imag(a32[i])), imag(a64[i]), 1e-3),
			"f32 vs f64 mismatch at %d: f32=%v f64=%v",
			i, a32[i], a64[i],
		)
	}
}

@test
test_c2c_auto_resolves_backend :: proc(t: ^testing.T) {
	plan: C2C_Plan
	err := c2c_plan_init_with_backend(&plan, 1024, .Auto)
	testing.expectf(t, err == .None, "c2c_plan_init_with_backend(.Auto) failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)

	testing.expectf(t, plan.backend == .Cooley_Tukey || plan.backend == .Split_Radix, "auto backend was not resolved: %v", plan.backend)
}

@test
test_c2c_roundtrip_split_radix :: proc(t: ^testing.T) {
	n := 64
	plan: C2C_Plan
	err := c2c_plan_init_with_backend(&plan, n, .Split_Radix)
	testing.expectf(t, err == .None, "c2c_plan_init_with_backend failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&plan)

	data, data_alloc_err := make([]complex128, n)
	orig, orig_alloc_err := make([]complex128, n)
	if data_alloc_err != .None || orig_alloc_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_alloc_err, orig_alloc_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for i in 0..<n {
		v := f64(i)
		data[i] = complex(math.sin(0.21*v)+0.05*v, math.cos(0.43*v)-0.02*v)
	}
	copy(orig, data)

	err = c2c_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_forward_in_place failed: %v", err)
	if err != .None {
		return
	}

	err = c2c_inverse_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_inverse_in_place failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], 2e-9), "split-radix roundtrip mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_forward_split_radix_matches_cooley_tukey :: proc(t: ^testing.T) {
	n := 128
	cooley: C2C_Plan
	split: C2C_Plan

	err := c2c_plan_init_with_backend(&cooley, n, .Cooley_Tukey)
	testing.expectf(t, err == .None, "cooley plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&cooley)

	err = c2c_plan_init_with_backend(&split, n, .Split_Radix)
	testing.expectf(t, err == .None, "split plan init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_destroy(&split)

	input, in_err := make([]complex128, n)
	a, a_err := make([]complex128, n)
	b, b_err := make([]complex128, n)
	if in_err != .None || a_err != .None || b_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v / %v", in_err, a_err, b_err)
		return
	}
	defer delete(input)
	defer delete(a)
	defer delete(b)

	for i in 0..<n {
		x := f64(i)
		input[i] = complex(
			math.sin(2.0*math.PI*x/17.0)+0.3*math.cos(2.0*math.PI*x/9.0),
			0.4*math.sin(2.0*math.PI*x/31.0)-0.2*math.cos(2.0*math.PI*x/13.0),
		)
	}
	copy(a, input)
	copy(b, input)

	err = c2c_forward_in_place(&cooley, a)
	testing.expectf(t, err == .None, "cooley forward failed: %v", err)
	if err != .None {
		return
	}

	err = c2c_forward_in_place(&split, b)
	testing.expectf(t, err == .None, "split forward failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(a[i], b[i], 2e-9), "backend mismatch at %d: cooley=%v split=%v", i, a[i], b[i])
	}
}

@test
test_r2c_c2r_roundtrip :: proc(t: ^testing.T) {
	n := 32
	plan: R2C_Plan
	err := r2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "r2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer r2c_plan_destroy(&plan)

	time_signal, in_err := make([]f64, n)
	freq_signal, fq_err := make([]complex128, n/2+1)
	recon, rc_err := make([]f64, n)
	if in_err != .None || fq_err != .None || rc_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v / %v", in_err, fq_err, rc_err)
		return
	}
	defer delete(time_signal)
	defer delete(freq_signal)
	defer delete(recon)

	for i in 0..<n {
		x := f64(i)
		time_signal[i] = math.sin(2.0*math.PI*x/8.0) + 0.35*math.cos(2.0*math.PI*x/5.0) + 0.01*x
	}

	err = r2c_forward(&plan, time_signal, freq_signal)
	testing.expectf(t, err == .None, "r2c_forward failed: %v", err)
	if err != .None {
		return
	}

	err = c2r_inverse(&plan, freq_signal, recon)
	testing.expectf(t, err == .None, "c2r_inverse failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_f64(recon[i], time_signal[i], EPS), "roundtrip mismatch at %d: got=%f want=%f", i, recon[i], time_signal[i])
	}
}

@test
test_r2c_c2r_roundtrip_non_power_of_two :: proc(t: ^testing.T) {
	n := 1000
	plan: R2C_Plan
	err := r2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "r2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer r2c_plan_destroy(&plan)
	testing.expectf(t, plan.uses_full_c2c, "expected full-c2c fallback for non-power-of-two n=%d", n)

	time_signal, in_err := make([]f64, n)
	freq_signal, fq_err := make([]complex128, r2c_output_len(n))
	recon, rc_err := make([]f64, n)
	if in_err != .None || fq_err != .None || rc_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v / %v", in_err, fq_err, rc_err)
		return
	}
	defer delete(time_signal)
	defer delete(freq_signal)
	defer delete(recon)

	for i in 0..<n {
		x := f64(i)
		time_signal[i] = math.sin(2.0*math.PI*x/17.0) + 0.25*math.cos(2.0*math.PI*x/23.0) + 0.001*x
	}

	err = r2c_forward(&plan, time_signal, freq_signal)
	testing.expectf(t, err == .None, "r2c_forward failed: %v", err)
	if err != .None {
		return
	}
	err = c2r_inverse(&plan, freq_signal, recon)
	testing.expectf(t, err == .None, "c2r_inverse failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_f64(recon[i], time_signal[i], 1e-8), "non-power-of-two roundtrip mismatch at %d: got=%f want=%f", i, recon[i], time_signal[i])
	}
}

@test
test_r2c_c2r_roundtrip_odd_length :: proc(t: ^testing.T) {
	n := 999
	plan: R2C_Plan
	err := r2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "r2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer r2c_plan_destroy(&plan)
	testing.expectf(t, plan.uses_full_c2c, "expected full-c2c fallback for odd n=%d", n)

	time_signal, in_err := make([]f64, n)
	freq_signal, fq_err := make([]complex128, r2c_output_len(n))
	recon, rc_err := make([]f64, n)
	if in_err != .None || fq_err != .None || rc_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v / %v", in_err, fq_err, rc_err)
		return
	}
	defer delete(time_signal)
	defer delete(freq_signal)
	defer delete(recon)

	for i in 0..<n {
		x := f64(i)
		time_signal[i] = 0.8*math.sin(2.0*math.PI*x/37.0) + 0.4*math.cos(2.0*math.PI*x/29.0) - 0.15*math.sin(2.0*math.PI*x/7.0)
	}

	err = r2c_forward(&plan, time_signal, freq_signal)
	testing.expectf(t, err == .None, "r2c_forward failed: %v", err)
	if err != .None {
		return
	}
	err = c2r_inverse(&plan, freq_signal, recon)
	testing.expectf(t, err == .None, "c2r_inverse failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_f64(recon[i], time_signal[i], 2e-8), "odd-length roundtrip mismatch at %d: got=%f want=%f", i, recon[i], time_signal[i])
	}
}

@test
test_r2c_impulse :: proc(t: ^testing.T) {
	n := 16
	plan: R2C_Plan
	err := r2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "r2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer r2c_plan_destroy(&plan)

	time_signal, in_err := make([]f64, n)
	freq_signal, fq_err := make([]complex128, n/2+1)
	if in_err != .None || fq_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", in_err, fq_err)
		return
	}
	defer delete(time_signal)
	defer delete(freq_signal)

	time_signal[0] = 1.0
	err = r2c_forward(&plan, time_signal, freq_signal)
	testing.expectf(t, err == .None, "r2c_forward failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<len(freq_signal) {
		testing.expectf(t, approx_eq_c128(freq_signal[i], complex(1.0, 0.0), EPS), "impulse mismatch at %d: %v", i, freq_signal[i])
	}
}

@test
test_r2c_external_scratch :: proc(t: ^testing.T) {
	n := 64
	plan: R2C_Plan
	err := r2c_plan_init(&plan, n)
	testing.expectf(t, err == .None, "r2c_plan_init failed: %v", err)
	if err != .None {
		return
	}
	defer r2c_plan_destroy(&plan)

	input, in_err := make([]f64, n)
	freq, fq_err := make([]complex128, n/2+1)
	output, out_err := make([]f64, n)
	scratch, sc_err := make([]complex128, n/2)
	if in_err != .None || fq_err != .None || out_err != .None || sc_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v / %v / %v", in_err, fq_err, out_err, sc_err)
		return
	}
	defer delete(input)
	defer delete(freq)
	defer delete(output)
	defer delete(scratch)

	for i in 0..<n {
		x := f64(i)
		input[i] = 0.2*math.sin(2.0*math.PI*x/11.0) + 0.7*math.cos(2.0*math.PI*x/7.0)
	}

	err = r2c_forward_with_scratch(&plan, input, freq, scratch)
	testing.expectf(t, err == .None, "r2c_forward_with_scratch failed: %v", err)
	if err != .None {
		return
	}

	err = c2r_inverse_with_scratch(&plan, freq, output, scratch)
	testing.expectf(t, err == .None, "c2r_inverse_with_scratch failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_f64(output[i], input[i], 2e-9), "external scratch mismatch at %d: got=%f want=%f", i, output[i], input[i])
	}
}

@test
test_r2c_random_roundtrip_many_sizes :: proc(t: ^testing.T) {
	rand.reset(0x0ddc0ffe)

	for exp in 1..<10 {
		n := 1
		for _ in 0..<exp {
			n *= 2
		}
		plan: R2C_Plan
		err := r2c_plan_init(&plan, n)
		testing.expectf(t, err == .None, "r2c_plan_init failed at n=%d: %v", n, err)
		if err != .None {
			return
		}

		time_signal, in_err := make([]f64, n)
		freq_signal, fq_err := make([]complex128, n/2+1)
		recon, rc_err := make([]f64, n)
		if in_err != .None || fq_err != .None || rc_err != .None {
			testing.expectf(t, false, "allocation failed at n=%d: %v / %v / %v", n, in_err, fq_err, rc_err)
			r2c_plan_destroy(&plan)
			return
		}

		for iter in 0..<5 {
			for i in 0..<n {
				time_signal[i] = rand.float64_range(-10.0, 10.0)
			}
			err = r2c_forward(&plan, time_signal, freq_signal)
			testing.expectf(t, err == .None, "r2c_forward failed at n=%d: %v", n, err)
			if err != .None {
				delete(time_signal)
				delete(freq_signal)
				delete(recon)
				r2c_plan_destroy(&plan)
				return
			}
			err = c2r_inverse(&plan, freq_signal, recon)
			testing.expectf(t, err == .None, "c2r_inverse failed at n=%d: %v", n, err)
			if err != .None {
				delete(time_signal)
				delete(freq_signal)
				delete(recon)
				r2c_plan_destroy(&plan)
				return
			}

			for i in 0..<n {
				testing.expectf(t, approx_eq_f64(recon[i], time_signal[i], 2e-9), "random roundtrip mismatch at n=%d i=%d got=%f want=%f", n, i, recon[i], time_signal[i])
			}
		}

		delete(time_signal)
		delete(freq_signal)
		delete(recon)
		r2c_plan_destroy(&plan)
	}
}

@test
test_c2c_2d_roundtrip :: proc(t: ^testing.T) {
	rows, cols := 8, 16
	plan: C2C_Plan_2D
	err := c2c_plan_2d_init(&plan, rows, cols)
	testing.expectf(t, err == .None, "c2c_plan_2d_init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_2d_destroy(&plan)

	n := rows * cols
	data, data_err := make([]complex128, n)
	orig, orig_err := make([]complex128, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for r in 0..<rows {
		for c in 0..<cols {
			x := f64(r*cols + c)
			v := complex(
				0.4*math.sin(2.0*math.PI*x/13.0)+0.1*f64(r),
				0.3*math.cos(2.0*math.PI*x/9.0)-0.05*f64(c),
			)
			data[r*cols+c] = v
		}
	}
	copy(orig, data)

	err = c2c_2d_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_2d_forward_in_place failed: %v", err)
	if err != .None {
		return
	}
	err = c2c_2d_inverse_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_2d_inverse_in_place failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], 3e-9), "2d roundtrip mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_2d_roundtrip_non_power_of_two :: proc(t: ^testing.T) {
	rows, cols := 7, 10
	plan: C2C_Plan_2D
	err := c2c_plan_2d_init(&plan, rows, cols)
	testing.expectf(t, err == .None, "c2c_plan_2d_init(non-power-of-two) failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_2d_destroy(&plan)

	n := rows * cols
	data, data_err := make([]complex128, n)
	orig, orig_err := make([]complex128, n)
	if data_err != .None || orig_err != .None {
		testing.expectf(t, false, "allocation failed: %v / %v", data_err, orig_err)
		return
	}
	defer delete(data)
	defer delete(orig)

	for r in 0..<rows {
		for c in 0..<cols {
			x := f64(r*cols + c)
			data[r*cols+c] = complex(
				0.5*math.sin(2.0*math.PI*x/19.0)+0.03*f64(r),
				0.3*math.cos(2.0*math.PI*x/11.0)-0.02*f64(c),
			)
		}
	}
	copy(orig, data)

	err = c2c_2d_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_2d_forward_in_place failed: %v", err)
	if err != .None {
		return
	}
	err = c2c_2d_inverse_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_2d_inverse_in_place failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], orig[i], 5e-8), "2d non-power-of-two roundtrip mismatch at %d: got=%v want=%v", i, data[i], orig[i])
	}
}

@test
test_c2c_2d_impulse :: proc(t: ^testing.T) {
	rows, cols := 8, 8
	plan: C2C_Plan_2D
	err := c2c_plan_2d_init(&plan, rows, cols)
	testing.expectf(t, err == .None, "c2c_plan_2d_init failed: %v", err)
	if err != .None {
		return
	}
	defer c2c_plan_2d_destroy(&plan)

	n := rows * cols
	data, data_err := make([]complex128, n)
	if data_err != .None {
		testing.expectf(t, false, "allocation failed: %v", data_err)
		return
	}
	defer delete(data)

	data[0] = complex(1.0, 0.0)
	err = c2c_2d_forward_in_place(&plan, data)
	testing.expectf(t, err == .None, "c2c_2d_forward_in_place failed: %v", err)
	if err != .None {
		return
	}

	for i in 0..<n {
		testing.expectf(t, approx_eq_c128(data[i], complex(1.0, 0.0), 2e-9), "2d impulse mismatch at %d: %v", i, data[i])
	}
}
