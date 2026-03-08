package fft

import "base:runtime"
import "core:math"

C2C_Plan_F32 :: struct {
	n:            int,
	log2_n:       int,
	backend:      Backend,
	uses_f64_fallback: bool,
	store_inverse_twiddles: bool,
	store_bitrev_table: bool,
	bitrev:       []u32,
	twiddles:     []complex64,
	twiddles_inv: []complex64,
	fallback64:   C2C_Plan,
	fallback_buf: []complex128,
	allocator:    runtime.Allocator,
	initialized:  bool,
}

resolve_backend_for_size_f32 :: #force_inline proc(n: int, requested: Backend, allocator: runtime.Allocator) -> Backend {
	b := resolve_backend_for_size(n, requested, allocator)
	if b == .Split_Radix {
		return .Cooley_Tukey
	}
	return b
}

c2c_plan_destroy_f32 :: proc(plan: ^C2C_Plan_F32) {
	if plan.bitrev != nil {
		delete(plan.bitrev, plan.allocator)
	}
	if plan.twiddles != nil {
		delete(plan.twiddles, plan.allocator)
	}
	if plan.twiddles_inv != nil {
		delete(plan.twiddles_inv, plan.allocator)
	}
	if plan.fallback_buf != nil {
		delete(plan.fallback_buf, plan.allocator)
	}
	if plan.fallback64.initialized {
		c2c_plan_destroy(&plan.fallback64)
	}
	plan^ = {}
}

c2c_plan_init_f32_with_backend :: proc(plan: ^C2C_Plan_F32, n: int, backend: Backend, allocator := context.allocator) -> Error {
	opts := C2C_Plan_Options{
		backend = backend,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 0,
		cooley_radix = 2,
	}
	return c2c_plan_init_f32_with_options(plan, n, opts, allocator)
}

c2c_plan_init_f32_with_options :: proc(plan: ^C2C_Plan_F32, n: int, options: C2C_Plan_Options, allocator := context.allocator) -> Error {
	if n < 1 {
		return .Invalid_Length
	}
	c2c_plan_destroy_f32(plan)
	if !is_power_of_two(n) {
		f64_err := c2c_plan_init_with_options(&plan.fallback64, n, options, allocator)
		if f64_err != .None {
			return f64_err
		}
		buf, buf_err := make([]complex128, n, allocator)
		if buf_err != .None {
			c2c_plan_destroy(&plan.fallback64)
			return .Allocation_Failed
		}
		plan.n = n
		plan.log2_n = 0
		plan.backend = plan.fallback64.backend
		plan.uses_f64_fallback = true
		plan.store_inverse_twiddles = false
		plan.store_bitrev_table = false
		plan.fallback_buf = buf
		plan.allocator = allocator
		plan.initialized = true
		return .None
	}

	resolved_backend := resolve_backend_for_size_f32(n, options.backend, allocator)
	log2_n := log2_exact(n)

	twiddles, twiddle_err := make([]complex64, n/2, allocator)
	if twiddle_err != .None {
		return .Allocation_Failed
	}
	should_store_inverse_twiddles := options.store_inverse_twiddles
	should_store_bitrev_table := options.store_bitrev_table
	twiddles_inv: []complex64
	if should_store_inverse_twiddles {
		inv_alloc_err: runtime.Allocator_Error
		twiddles_inv, inv_alloc_err = make([]complex64, n/2, allocator)
		if inv_alloc_err != .None {
			delete(twiddles, allocator)
			return .Allocation_Failed
		}
	}
	bitrev: []u32
	if should_store_bitrev_table {
		bitrev_err: runtime.Allocator_Error
		bitrev, bitrev_err = make([]u32, n, allocator)
		if bitrev_err != .None {
			delete(twiddles, allocator)
			if twiddles_inv != nil {
				delete(twiddles_inv, allocator)
			}
			return .Allocation_Failed
		}
	}

	#no_bounds_check for k in 0..<len(twiddles) {
		angle := -2.0 * math.PI * f64(k) / f64(n)
		s, c := math.sincos(angle)
		w := complex(f32(c), f32(s))
		twiddles[k] = w
		if should_store_inverse_twiddles {
			twiddles_inv[k] = conj(w)
		}
	}
	if should_store_bitrev_table {
		#no_bounds_check for i in 0..<n {
			bitrev[i] = reverse_bits_u32(u32(i), log2_n)
		}
	}

	plan.n = n
	plan.log2_n = log2_n
	plan.backend = resolved_backend
	plan.store_inverse_twiddles = should_store_inverse_twiddles
	plan.store_bitrev_table = should_store_bitrev_table
	plan.bitrev = bitrev
	plan.twiddles = twiddles
	plan.twiddles_inv = twiddles_inv
	plan.allocator = allocator
	plan.initialized = true
	return .None
}

c2c_plan_init_f32 :: proc(plan: ^C2C_Plan_F32, n: int, allocator := context.allocator) -> Error {
	return c2c_plan_init_f32_with_backend(plan, n, .Auto, allocator)
}

c2c_plan_init_f32_low_ram :: proc(plan: ^C2C_Plan_F32, n: int, backend := Backend.Auto, allocator := context.allocator) -> Error {
	opts := C2C_Plan_Options{
		backend = backend,
		store_inverse_twiddles = false,
		store_bitrev_table = false,
		threads = 0,
		cooley_radix = 2,
	}
	return c2c_plan_init_f32_with_options(plan, n, opts, allocator)
}

c2c_plan_estimate_bytes_f32 :: proc(n: int, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	if n < 1 {
		return 0
	}
	if !is_power_of_two(n) {
		base := c2c_plan_estimate_bytes(n, backend, store_inverse_twiddles, store_bitrev_table)
		if base <= 0 {
			return 0
		}
		return base + n*size_of(complex128)
	}
	resolved_backend := resolve_backend_for_size_f32(n, backend, context.allocator)
	bytes := 0
	bytes += (n / 2) * size_of(complex64)
	if store_inverse_twiddles {
		bytes += (n / 2) * size_of(complex64)
	}
	if store_bitrev_table {
		bytes += n * size_of(u32)
	}
	_ = resolved_backend
	return bytes
}

bit_reverse_permute_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) {
	n := plan.n
	if plan.bitrev != nil {
		#no_bounds_check for i in 0..<n {
			j := int(plan.bitrev[i])
			if j > i {
				data[i], data[j] = data[j], data[i]
			}
		}
		return
	}

	j := 0
	for i := 1; i < n-1; i += 1 {
		bit := n >> 1
		for (j & bit) != 0 {
			j &= ~bit
			bit >>= 1
		}
		j |= bit
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
	}
}

c2c_forward_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.n {
		return .Size_Mismatch
	}
	if len(data) <= 1 {
		return .None
	}
	if plan.uses_f64_fallback {
		#no_bounds_check for i in 0..<plan.n {
			v := data[i]
			plan.fallback_buf[i] = complex(f64(real(v)), f64(imag(v)))
		}
		if err := c2c_forward_in_place(&plan.fallback64, plan.fallback_buf); err != .None {
			return err
		}
		#no_bounds_check for i in 0..<plan.n {
			v := plan.fallback_buf[i]
			data[i] = complex(f32(real(v)), f32(imag(v)))
		}
		return .None
	}

	n := plan.n
	bit_reverse_permute_in_place_f32(plan, data)

	for i := 0; i < n; i += 2 {
		a := data[i]
		b := data[i+1]
		data[i] = a + b
		data[i+1] = a - b
	}

	if n == 2 {
		return .None
	}

	len := 4
	for len <= n {
		half := len / 2
		stride := n / len
		for base := 0; base < n; base += len {
			#no_bounds_check {
				u0 := data[base]
				v0 := data[base+half]
				data[base] = u0 + v0
				data[base+half] = u0 - v0
			}
			#no_bounds_check for k in 1..<half {
				u := data[base+k]
				v := data[base+k+half] * plan.twiddles[k*stride]
				data[base+k] = u + v
				data[base+k+half] = u - v
			}
		}
		len <<= 1
	}

	return .None
}

c2c_inverse_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.n {
		return .Size_Mismatch
	}
	if len(data) <= 1 {
		return .None
	}
	if plan.uses_f64_fallback {
		#no_bounds_check for i in 0..<plan.n {
			v := data[i]
			plan.fallback_buf[i] = complex(f64(real(v)), f64(imag(v)))
		}
		if err := c2c_inverse_in_place(&plan.fallback64, plan.fallback_buf); err != .None {
			return err
		}
		#no_bounds_check for i in 0..<plan.n {
			v := plan.fallback_buf[i]
			data[i] = complex(f32(real(v)), f32(imag(v)))
		}
		return .None
	}

	n := plan.n
	bit_reverse_permute_in_place_f32(plan, data)

	for i := 0; i < n; i += 2 {
		a := data[i]
		b := data[i+1]
		data[i] = a + b
		data[i+1] = a - b
	}

	if n > 2 {
		len := 4
		if plan.twiddles_inv != nil {
			for len <= n {
				half := len / 2
				stride := n / len
				for base := 0; base < n; base += len {
					#no_bounds_check {
						u0 := data[base]
						v0 := data[base+half]
						data[base] = u0 + v0
						data[base+half] = u0 - v0
					}
					#no_bounds_check for k in 1..<half {
						u := data[base+k]
						v := data[base+k+half] * plan.twiddles_inv[k*stride]
						data[base+k] = u + v
						data[base+k+half] = u - v
					}
				}
				len <<= 1
			}
		} else {
			for len <= n {
				half := len / 2
				stride := n / len
				for base := 0; base < n; base += len {
					#no_bounds_check {
						u0 := data[base]
						v0 := data[base+half]
						data[base] = u0 + v0
						data[base+half] = u0 - v0
					}
					#no_bounds_check for k in 1..<half {
						u := data[base+k]
						v := data[base+k+half] * conj(plan.twiddles[k*stride])
						data[base+k] = u + v
						data[base+k+half] = u - v
					}
				}
				len <<= 1
			}
		}
	}

	inv_n := f32(1.0 / f64(n))
	#no_bounds_check for i in 0..<n {
		data[i] *= complex(inv_n, f32(0.0))
	}

	return .None
}

c2c_forward_f32 :: proc(plan: ^C2C_Plan_F32, input: []complex64, output: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.n {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_forward_in_place_f32(plan, output)
}

c2c_inverse_f32 :: proc(plan: ^C2C_Plan_F32, input: []complex64, output: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.n {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_inverse_in_place_f32(plan, output)
}
