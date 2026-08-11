package fft

import "base:runtime"
import "core:math"

R2C_Plan_F32 :: struct {
	n:              int,
	half_n:         int,
	uses_full_c2c:  bool,
	c2c:            C2C_Plan_F32,
	twiddles:       []complex64,
	twiddles_inv:   []complex64,
	scratch:        []complex64,
	allocator:      runtime.Allocator,
	initialized:    bool,
}

r2c_plan_destroy_f32 :: proc(plan: ^R2C_Plan_F32) {
	c2c_plan_destroy_f32(&plan.c2c)
	if plan.twiddles != nil {
		delete(plan.twiddles, plan.allocator)
	}
	if plan.twiddles_inv != nil {
		delete(plan.twiddles_inv, plan.allocator)
	}
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	plan^ = {}
}

r2c_plan_init_f32 :: proc(plan: ^R2C_Plan_F32, n: int, allocator := context.allocator) -> Error {
	if n < 2 {
		return .Invalid_Length
	}
	r2c_plan_destroy_f32(plan)

	half_n := n / 2
	if is_power_of_two(n) {
		if err := c2c_plan_init_f32(&plan.c2c, half_n, allocator=allocator); err != .None {
			return err
		}

		twiddles, twiddle_err := make([]complex64, half_n/2+1, allocator)
		if twiddle_err != .None {
			r2c_plan_destroy_f32(plan)
			return .Allocation_Failed
		}
		twiddles_inv, twiddle_inv_err := make([]complex64, half_n/2+1, allocator)
		if twiddle_inv_err != .None {
			delete(twiddles, allocator)
			r2c_plan_destroy_f32(plan)
			return .Allocation_Failed
		}

		#no_bounds_check for k in 0..<len(twiddles) {
			angle := -2.0 * math.PI * f64(k) / f64(n)
			s, c := math.sincos(angle)
			twiddles[k] = complex(f32(c), f32(s))
			twiddles_inv[k] = complex(f32(c), f32(-s))
		}

		plan.twiddles = twiddles
		plan.twiddles_inv = twiddles_inv
		plan.uses_full_c2c = false
	} else {
		if err := c2c_plan_init_f32(&plan.c2c, n, allocator=allocator); err != .None {
			return err
		}
		scratch, scratch_err := make([]complex64, n, allocator)
		if scratch_err != .None {
			r2c_plan_destroy_f32(plan)
			return .Allocation_Failed
		}
		plan.scratch = scratch
		plan.uses_full_c2c = true
	}

	plan.n = n
	plan.half_n = half_n
	plan.allocator = allocator
	plan.initialized = true
	return .None
}

r2c_forward_f32_with_scratch :: proc(
	plan: ^R2C_Plan_F32,
	input: []f32,
	output: []complex64,
	scratch: []complex64,
) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.half_n+1 {
		return .Size_Mismatch
	}
	if plan.uses_full_c2c {
		if len(scratch) != plan.n {
			return .Size_Mismatch
		}
		#no_bounds_check for i in 0..<plan.n {
			scratch[i] = complex(input[i], f32(0.0))
		}
		if err := c2c_forward_in_place_f32(&plan.c2c, scratch); err != .None {
			return err
		}
		copy(output, scratch[:plan.half_n+1])
		return .None
	}

	// Pack the real input into the complex output buffer and transform N/2 values.
	work := output[:plan.half_n]
	copy(([^]f32)(raw_data(work))[:2*plan.half_n], input)
	if err := c2c_forward_in_place_f32(&plan.c2c, work); err != .None {
		return err
	}

	half := f32(0.5)
	a0 := work[0]
	output[0] = complex(real(a0)+imag(a0), f32(0.0))
	output[plan.half_n] = complex(real(a0)-imag(a0), f32(0.0))

	quarter := plan.half_n / 2
	for k := 1; k < quarter; k += 1 {
		mirror := plan.half_n - k
		a := work[k]
		b := work[mirror]
		w := plan.twiddles[k]
		ar, ai := real(a), imag(a)
		br, bi := real(b), imag(b)
		wr, wi := real(w), imag(w)
		sum_r := ar + br
		sum_i := ai + bi
		diff_r := ar - br
		diff_i := ai - bi
		re_term := half * (wr*sum_i + wi*diff_r)
		im_base := half * (wi*sum_i - wr*diff_r)
		im_delta := half * diff_i
		output[k] = complex(half*sum_r+re_term, im_base+im_delta)
		output[mirror] = complex(half*sum_r-re_term, im_base-im_delta)
	}
	if plan.half_n > 1 {
		k := quarter
		a := work[k]
		w := plan.twiddles[k]
		output[k] = complex(real(a)+real(w)*imag(a), imag(w)*imag(a))
	}

	return .None
}

r2c_forward_f32 :: proc(plan: ^R2C_Plan_F32, input: []f32, output: []complex64) -> Error {
	return r2c_forward_f32_with_scratch(plan, input, output, plan.scratch)
}

c2r_inverse_f32_with_scratch :: proc(
	plan: ^R2C_Plan_F32,
	input: []complex64,
	output: []f32,
	scratch: []complex64,
) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.half_n+1 || len(output) != plan.n {
		return .Size_Mismatch
	}
	if plan.uses_full_c2c {
		if len(scratch) != plan.n {
			return .Size_Mismatch
		}
		#no_bounds_check for i in 0..<plan.n {
			scratch[i] = complex(f32(0.0), f32(0.0))
		}
		#no_bounds_check for k in 0..=plan.half_n {
			scratch[k] = input[k]
		}
		#no_bounds_check for k in 1..=plan.half_n {
			mirror := plan.n - k
			if mirror != k {
				scratch[mirror] = conj(input[k])
			} else {
				scratch[k] = complex(real(input[k]), f32(0.0))
			}
		}
		if err := c2c_inverse_in_place_f32(&plan.c2c, scratch); err != .None {
			return err
		}
		#no_bounds_check for i in 0..<plan.n {
			output[i] = real(scratch[i])
		}
		return .None
	}

	work := ([^]complex64)(raw_data(output))[:plan.half_n]
	x0 := input[0]
	xn2 := input[plan.half_n]
	work[0] = complex(
		half_f32*(real(x0)+real(xn2)),
		half_f32*(real(x0)-real(xn2)),
	)

	half := f32(0.5)
	quarter := plan.half_n / 2
	for k := 1; k < quarter; k += 1 {
		mirror := plan.half_n - k
		c := input[k]
		w := plan.twiddles_inv[k]
		m := input[mirror]
		cr, ci := real(c), imag(c)
		mr, mi := real(m), imag(m)
		wr, wi := real(w), imag(w)
		sum_r := cr + mr
		sum_i := ci + mi
		diff_r := cr - mr
		diff_i := ci - mi
		re_term := half * (wi*diff_r + wr*sum_i)
		im_base := half * (wr*diff_r - wi*sum_i)
		im_delta := half * diff_i
		work[k] = complex(half*sum_r-re_term, im_base+im_delta)
		work[mirror] = complex(half*sum_r+re_term, im_base-im_delta)
	}
	if plan.half_n > 1 {
		k := quarter
		c := input[k]
		w := plan.twiddles_inv[k]
		work[k] = complex(real(c)-real(w)*imag(c), -imag(w)*imag(c))
	}

	return c2c_inverse_in_place_f32(&plan.c2c, work)
}

c2r_inverse_f32 :: proc(plan: ^R2C_Plan_F32, input: []complex64, output: []f32) -> Error {
	return c2r_inverse_f32_with_scratch(plan, input, output, plan.scratch)
}

half_f32 :: f32(0.5)
