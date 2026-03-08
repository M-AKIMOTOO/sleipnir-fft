package fft

import "base:runtime"
import "core:math"

next_power_of_two :: #force_inline proc(v: int) -> int {
	if v <= 1 {
		return 1
	}
	p := 1
	for p < v {
		p <<= 1
		if p <= 0 {
			return 0
		}
	}
	return p
}

c2c_plan_init_bluestein :: proc(plan: ^C2C_Plan, n: int, options: C2C_Plan_Options, allocator: runtime.Allocator) -> Error {
	c2c_plan_destroy(plan)

	m := next_power_of_two(2*n - 1)
	if m <= 0 {
		return .Invalid_Length
	}

	chirp, chirp_err := make([]complex128, n, allocator)
	if chirp_err != .None {
		return .Allocation_Failed
	}
	work, work_err := make([]complex128, m, allocator)
	if work_err != .None {
		delete(chirp, allocator)
		return .Allocation_Failed
	}
	b_fft, b_err := make([]complex128, m, allocator)
	if b_err != .None {
		delete(work, allocator)
		delete(chirp, allocator)
		return .Allocation_Failed
	}

	inv_n := 1.0 / f64(n)
	#no_bounds_check for i in 0..<n {
		t := f64(i)
		angle := math.PI * t * t * inv_n
		s, c := math.sincos(angle)
		w_fwd := complex(c, -s) // exp(-i*pi*i^2/N)
		w_inv := complex(c, s)  // exp(+i*pi*i^2/N)
		chirp[i] = w_fwd
		b_fft[i] = w_inv
		if i > 0 {
			b_fft[m-i] = w_inv
		}
	}

	conv_plan := new(C2C_Plan, allocator)
	if conv_plan == nil {
		delete(b_fft, allocator)
		delete(work, allocator)
		delete(chirp, allocator)
		return .Allocation_Failed
	}
	conv_opts := C2C_Plan_Options{
		backend = .Cooley_Tukey,
		store_inverse_twiddles = options.store_inverse_twiddles,
		store_bitrev_table = options.store_bitrev_table,
		threads = options.threads,
		cooley_radix = options.cooley_radix,
	}
	if err := c2c_plan_init_with_options(conv_plan, m, conv_opts, allocator); err != .None {
		free(conv_plan, allocator)
		delete(b_fft, allocator)
		delete(work, allocator)
		delete(chirp, allocator)
		return err
	}

	if err := c2c_forward_in_place(conv_plan, b_fft); err != .None {
		c2c_plan_destroy(conv_plan)
		free(conv_plan, allocator)
		delete(b_fft, allocator)
		delete(work, allocator)
		delete(chirp, allocator)
		return err
	}

	plan.n = n
	plan.log2_n = 0
	plan.backend = .Cooley_Tukey
	plan.uses_bluestein = true
	plan.store_inverse_twiddles = options.store_inverse_twiddles
	plan.store_bitrev_table = options.store_bitrev_table
	plan.num_threads = resolve_thread_count(options.threads, n)
	plan.cooley_radix = 0
	plan.bluestein_chirp = chirp
	plan.bluestein_m = work
	plan.bluestein_b_fft = b_fft
	plan.bluestein_conv_plan = conv_plan
	plan.allocator = allocator
	plan.initialized = true
	return .None
}

bluestein_forward_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	if !plan.uses_bluestein || plan.bluestein_conv_plan == nil {
		return .Plan_Not_Initialized
	}
	n := plan.n
	m := len(plan.bluestein_m)
	work := plan.bluestein_m
	chirp := plan.bluestein_chirp
	b_fft := plan.bluestein_b_fft
	conv := plan.bluestein_conv_plan

	#no_bounds_check for i in 0..<m {
		work[i] = 0
	}
	#no_bounds_check for i in 0..<n {
		work[i] = data[i] * chirp[i]
	}
	if err := c2c_forward_in_place(conv, work); err != .None {
		return err
	}
	#no_bounds_check for i in 0..<m {
		work[i] *= b_fft[i]
	}
	if err := c2c_inverse_in_place(conv, work); err != .None {
		return err
	}
	#no_bounds_check for i in 0..<n {
		data[i] = work[i] * chirp[i]
	}
	return .None
}

bluestein_inverse_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	#no_bounds_check for i in 0..<n {
		data[i] = conj(data[i])
	}
	if err := bluestein_forward_in_place(plan, data); err != .None {
		return err
	}
	inv_n := 1.0 / f64(n)
	#no_bounds_check for i in 0..<n {
		data[i] = conj(data[i]) * complex(inv_n, 0.0)
	}
	return .None
}
