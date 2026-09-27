package fft

import "base:intrinsics"
import "base:runtime"
import "core:simd"
import "core:math"
simd_cmul2_f32x4 :: #force_inline proc(a, b: simd.f32x4) -> simd.f32x4 {
	ar := simd.shuffle(a, a, 0, 0, 2, 2)
	ai := simd.shuffle(a, a, 1, 1, 3, 3)
	br := simd.shuffle(b, b, 0, 0, 2, 2)
	bi := simd.shuffle(b, b, 1, 1, 3, 3)
	re := simd.fused_mul_add(ar, br, -simd.mul(ai, bi))
	im := simd.fused_mul_add(ar, bi, simd.mul(ai, br))
	return simd.shuffle(re, im, 0, 4, 2, 6)
}
simd_cmul4_f32x8 :: #force_inline proc(a, b: simd.f32x8) -> simd.f32x8 {
    ar := simd.shuffle(a, a, 0, 0, 2, 2, 4, 4, 6, 6)
    ai := simd.shuffle(a, a, 1, 1, 3, 3, 5, 5, 7, 7)
    br := simd.shuffle(b, b, 0, 0, 2, 2, 4, 4, 6, 6)
    bi := simd.shuffle(b, b, 1, 1, 3, 3, 5, 5, 7, 7)
    re := simd.fused_mul_add(ar, br, -simd.mul(ai, bi))
    im := simd.fused_mul_add(ar, bi, simd.mul(ai, br))
    return simd.shuffle(re, im, 0, 8, 2, 10, 4, 12, 6, 14)
}

simd_scale_complex64x4 :: #force_inline proc(v: simd.f32x8, scale: f32) -> simd.f32x8 {
    return simd.mul(v, simd.f32x8{scale, scale, scale, scale, scale, scale, scale, scale})
}

simd_radix4_butterfly_f32x8 :: #force_inline proc(a, b, c, d, w1, w2, w3, rotation: simd.f32x8) -> (y0, y1, y2, y3: simd.f32x8) {
    bv := simd_cmul4_f32x8(b, w1)
    cv := simd_cmul4_f32x8(c, w2)
    dv := simd_cmul4_f32x8(d, w3)
    t0 := simd.add(a, cv)
    t1 := simd.sub(a, cv)
    t2 := simd.add(bv, dv)
    bd := simd.sub(bv, dv)
    t3 := simd.mul(simd.shuffle(bd, bd, 1, 0, 3, 2, 5, 4, 7, 6), rotation)
    y0 = simd.add(t0, t2)
    y1 = simd.add(t1, t3)
    y2 = simd.sub(t0, t2)
    y3 = simd.sub(t1, t3)
    return
}

simd_scale_complex64x2 :: #force_inline proc(v: simd.f32x4, scale: f32) -> simd.f32x4 {
	return simd.mul(v, simd.f32x4{scale, scale, scale, scale})
}


Radix4_F32_SIMD_Twiddle :: struct {
    w1: simd.f32x8,
    w2: simd.f32x8,
    w3: simd.f32x8,
}

Radix4_F32_SIMD_Stage :: struct {
    len: int,
    tw: []Radix4_F32_SIMD_Twiddle,
}

C2C_Plan_F32 :: struct {
	n:            int,
	log2_n:       int,
	backend:      Backend,
	cooley_radix: int,
	uses_f64_fallback: bool,
	store_inverse_twiddles: bool,
	store_bitrev_table: bool,
	bitrev:       []u32,
	twiddles:     []complex64,
	twiddles_inv: []complex64,
	radix4_simd_stages: []Radix4_F32_SIMD_Stage,
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

pack_complex64x4_f32x8 :: #force_inline proc(a, b, c, d: complex64) -> simd.f32x8 {
    return simd.f32x8{real(a), imag(a), real(b), imag(b), real(c), imag(c), real(d), imag(d)}
}

make_radix4_f32_simd_stages :: proc(n, log2_n: int, twiddles: []complex64, enabled: bool, allocator: runtime.Allocator) -> ([]Radix4_F32_SIMD_Stage, Error) {
    if !enabled || n < 4096 { return nil, .None }
    stage_count := log2_n / 2 - 1
    stages, stages_err := make([]Radix4_F32_SIMD_Stage, stage_count, allocator)
    if stages_err != .None { return nil, .Allocation_Failed }
    stage_len := 16
    for si in 0..<stage_count {
        quarter := stage_len / 4
        pair_count := quarter / 4
        packed, packed_err := make([]Radix4_F32_SIMD_Twiddle, pair_count, allocator)
        if packed_err != .None {
            for old in stages { if old.tw != nil { delete(old.tw, allocator) } }
            delete(stages, allocator)
            return nil, .Allocation_Failed
        }
        stride := n / stage_len
        half_n := n / 2
        for p in 0..<pair_count {
            k := p * 4
            idx1 := k * stride
            idx2 := k * stride * 2
            idx3 := k * stride * 3
            w1 := pack_complex64x4_f32x8(twiddles[idx1], twiddles[idx1+stride], twiddles[idx1+2*stride], twiddles[idx1+3*stride])
            w2 := pack_complex64x4_f32x8(twiddles[idx2], twiddles[idx2+2*stride], twiddles[idx2+4*stride], twiddles[idx2+6*stride])
            w3a := twiddles[idx3] if idx3 < half_n else -twiddles[idx3-half_n]
            w3b := twiddles[idx3+3*stride] if idx3+3*stride < half_n else -twiddles[idx3+3*stride-half_n]
            w3c := twiddles[idx3+6*stride] if idx3+6*stride < half_n else -twiddles[idx3+6*stride-half_n]
            w3d := twiddles[idx3+9*stride] if idx3+9*stride < half_n else -twiddles[idx3+9*stride-half_n]
            packed[p] = Radix4_F32_SIMD_Twiddle{
                w1 = w1,
                w2 = w2,
                w3 = pack_complex64x4_f32x8(w3a, w3b, w3c, w3d),
            }
        }
        stages[si] = Radix4_F32_SIMD_Stage{len = stage_len, tw = packed}
        stage_len *= 4
    }
    return stages, .None
}

c2c_plan_destroy_f32 :: proc(plan: ^C2C_Plan_F32) {
	if plan.radix4_simd_stages != nil {
		for stage in plan.radix4_simd_stages {
			if stage.tw != nil { delete(stage.tw, plan.allocator) }
		}
		delete(plan.radix4_simd_stages, plan.allocator)
	}
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
		cooley_radix = 0,
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
	cooley_radix := resolve_cooley_radix(n, log2_n, options.cooley_radix, 1)
	if options.cooley_radix == 0 && n < 16384 && (log2_n & 1) == 0 { cooley_radix = 2 }

	twiddles, twiddle_err := make([]complex64, n/2, allocator)
	if twiddle_err != .None {
		return .Allocation_Failed
	}
	should_store_inverse_twiddles := options.store_inverse_twiddles
	should_store_bitrev_table := options.store_bitrev_table && cooley_radix == 2
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
	if should_store_bitrev_table && cooley_radix == 2 {
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
	if should_store_bitrev_table && cooley_radix == 2 {
		#no_bounds_check for i in 0..<n {
			bitrev[i] = reverse_bits_u32(u32(i), log2_n)
		}
	}

	radix4_stages, radix4_stages_err := make_radix4_f32_simd_stages(n, log2_n, twiddles, cooley_radix == 4 && should_store_inverse_twiddles, allocator)
	if radix4_stages_err != .None {
		if bitrev != nil { delete(bitrev, allocator) }
		if twiddles_inv != nil { delete(twiddles_inv, allocator) }
		delete(twiddles, allocator)
		return radix4_stages_err
	}
	plan.n = n
	plan.log2_n = log2_n
	plan.cooley_radix = cooley_radix
	plan.backend = resolved_backend
	plan.store_inverse_twiddles = should_store_inverse_twiddles
	plan.store_bitrev_table = should_store_bitrev_table
	plan.bitrev = bitrev
	plan.twiddles = twiddles
	plan.twiddles_inv = twiddles_inv
	plan.radix4_simd_stages = radix4_stages
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
		cooley_radix = 0,
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
	log2_n := log2_exact(n)
	cooley_radix := resolve_cooley_radix(n, log2_n, 0, 1)
	if n < 16384 && (log2_n & 1) == 0 { cooley_radix = 2 }
	if store_bitrev_table && cooley_radix == 2 {
		bytes += n * size_of(u32)
	}
	if cooley_radix == 4 && n >= 4096 {
		stage_len := 16
		for stage_len <= n {
			bytes += (stage_len / 16) * 3 * size_of(simd.f32x8)
			stage_len *= 4
		}
	}
	_ = resolved_backend
	return bytes
}

bit_reverse_permute_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) {
	n := plan.n
	if plan.radix4_simd_stages != nil {
		for stage in plan.radix4_simd_stages {
			if stage.tw != nil { delete(stage.tw, plan.allocator) }
		}
		delete(plan.radix4_simd_stages, plan.allocator)
	}
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
    if !plan.initialized { return .Plan_Not_Initialized }
    if len(data) != plan.n { return .Size_Mismatch }
    if len(data) <= 1 { return .None }
    if plan.uses_f64_fallback {
        #no_bounds_check for i in 0..<plan.n {
            v := data[i]
            plan.fallback_buf[i] = complex(f64(real(v)), f64(imag(v)))
        }
        if err := c2c_forward_in_place(&plan.fallback64, plan.fallback_buf); err != .None { return err }
        #no_bounds_check for i in 0..<plan.n {
            v := plan.fallback_buf[i]
            data[i] = complex(f32(real(v)), f32(imag(v)))
        }
        return .None
    }
    if plan.cooley_radix == 4 { return cooley_tukey_forward_radix4_in_place_f32(plan, data) }

    n := plan.n
    bit_reverse_permute_in_place_f32(plan, data)
    i := 0
    when FFT_USE_SIMD_KERNELS {
        for ; i+3 < n && n >= 4096; i += 4 {
            v := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i]))
            u := simd.shuffle(v, v, 0, 1, 0, 1, 4, 5, 4, 5)
            w := simd.shuffle(v, v, 2, 3, 2, 3, 6, 7, 6, 7)
            s := simd.add(u, w)
            d := simd.sub(u, w)
            out := simd.shuffle(s, d, 0, 1, 8, 9, 4, 5, 12, 13)
            intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i]), out)
        }
        for ; i+1 < n; i += 2 {
            v := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i]))
            u := simd.shuffle(v, v, 0, 1, 0, 1)
            w := simd.shuffle(v, v, 2, 3, 2, 3)
            intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i]), simd.shuffle(u+w, u-w, 0, 1, 4, 5))
        }
    }
    for ; i < n; i += 2 {
        a := data[i]; b := data[i+1]
        data[i] = a + b; data[i+1] = a - b
    }
    if n == 2 { return .None }
    len := 4
    for len <= n {
        half := len / 2
        stride := n / len
        for base := 0; base < n; base += len {
            #no_bounds_check {
                u0 := data[base]; v0 := data[base+half]
                data[base] = u0 + v0; data[base+half] = u0 - v0
            }
            k := 1
            when FFT_USE_SIMD_KERNELS {
                for ; k+3 < half && n >= 4096; k += 4 {
                    iu := base + k; iv := iu + half
                    u := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[iu]))
                    v := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[iv]))
                    t0 := plan.twiddles[k*stride]; t1 := plan.twiddles[(k+1)*stride]
                    t2 := plan.twiddles[(k+2)*stride]; t3 := plan.twiddles[(k+3)*stride]
                    w := simd.f32x8{real(t0), imag(t0), real(t1), imag(t1), real(t2), imag(t2), real(t3), imag(t3)}
                    vw := simd_cmul4_f32x8(v, w)
                    intrinsics.unaligned_store(cast(^simd.f32x8)(&data[iu]), u+vw)
                    intrinsics.unaligned_store(cast(^simd.f32x8)(&data[iv]), u-vw)
                }
                for ; k+1 < half; k += 2 {
                    iu := base + k; iv := iu + half
                    u := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[iu]))
                    v := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[iv]))
                    t0 := plan.twiddles[k*stride]; t1 := plan.twiddles[(k+1)*stride]
                    w := simd.f32x4{real(t0), imag(t0), real(t1), imag(t1)}
                    vw := simd_cmul2_f32x4(v, w)
                    intrinsics.unaligned_store(cast(^simd.f32x4)(&data[iu]), u+vw)
                    intrinsics.unaligned_store(cast(^simd.f32x4)(&data[iv]), u-vw)
                }
            }
            #no_bounds_check for ; k < half; k += 1 {
                u := data[base+k]
                v := data[base+k+half] * plan.twiddles[k*stride]
                data[base+k] = u + v; data[base+k+half] = u - v
            }
        }
        len <<= 1
    }
    return .None
}
c2c_inverse_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) -> Error {
    if !plan.initialized { return .Plan_Not_Initialized }
    if len(data) != plan.n { return .Size_Mismatch }
    if len(data) <= 1 { return .None }
    if plan.uses_f64_fallback {
        #no_bounds_check for i in 0..<plan.n {
            v := data[i]
            plan.fallback_buf[i] = complex(f64(real(v)), f64(imag(v)))
        }
        if err := c2c_inverse_in_place(&plan.fallback64, plan.fallback_buf); err != .None { return err }
        #no_bounds_check for i in 0..<plan.n {
            v := plan.fallback_buf[i]
            data[i] = complex(f32(real(v)), f32(imag(v)))
        }
        return .None
    }
    if plan.cooley_radix == 4 { return cooley_tukey_inverse_radix4_in_place_f32(plan, data) }

    n := plan.n
    bit_reverse_permute_in_place_f32(plan, data)
    i := 0
    when FFT_USE_SIMD_KERNELS {
        for ; i+3 < n && n >= 4096; i += 4 {
            v := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i]))
            u := simd.shuffle(v, v, 0, 1, 0, 1, 4, 5, 4, 5)
            w := simd.shuffle(v, v, 2, 3, 2, 3, 6, 7, 6, 7)
            s := simd.add(u, w)
            d := simd.sub(u, w)
            out := simd.shuffle(s, d, 0, 1, 8, 9, 4, 5, 12, 13)
            intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i]), out)
        }
        for ; i+1 < n; i += 2 {
            v := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i]))
            u := simd.shuffle(v, v, 0, 1, 0, 1)
            w := simd.shuffle(v, v, 2, 3, 2, 3)
            intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i]), simd.shuffle(u+w, u-w, 0, 1, 4, 5))
        }
    }
    for ; i < n; i += 2 {
        a := data[i]; b := data[i+1]
        data[i] = a + b; data[i+1] = a - b
    }
    if n > 2 {
        len := 4
        for len <= n {
            half := len / 2
            stride := n / len
            for base := 0; base < n; base += len {
                #no_bounds_check {
                    u0 := data[base]; v0 := data[base+half]
                    data[base] = u0 + v0; data[base+half] = u0 - v0
                }
                k := 1
                when FFT_USE_SIMD_KERNELS {
                    for ; k+3 < half && n >= 4096; k += 4 {
                        iu := base + k; iv := iu + half
                        u := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[iu]))
                        v := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[iv]))
                        t0 := plan.twiddles[k*stride]; t1 := plan.twiddles[(k+1)*stride]
                        t2 := plan.twiddles[(k+2)*stride]; t3 := plan.twiddles[(k+3)*stride]
                        if plan.twiddles_inv != nil {
                            t0 = plan.twiddles_inv[k*stride]; t1 = plan.twiddles_inv[(k+1)*stride]
                            t2 = plan.twiddles_inv[(k+2)*stride]; t3 = plan.twiddles_inv[(k+3)*stride]
                        } else { t0, t1, t2, t3 = conj(t0), conj(t1), conj(t2), conj(t3) }
                        w := simd.f32x8{real(t0), imag(t0), real(t1), imag(t1), real(t2), imag(t2), real(t3), imag(t3)}
                        vw := simd_cmul4_f32x8(v, w)
                        intrinsics.unaligned_store(cast(^simd.f32x8)(&data[iu]), u+vw)
                        intrinsics.unaligned_store(cast(^simd.f32x8)(&data[iv]), u-vw)
                    }
                    for ; k+1 < half; k += 2 {
                        iu := base + k; iv := iu + half
                        u := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[iu]))
                        v := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[iv]))
                        t0 := plan.twiddles[k*stride]; t1 := plan.twiddles[(k+1)*stride]
                        if plan.twiddles_inv != nil { t0 = plan.twiddles_inv[k*stride]; t1 = plan.twiddles_inv[(k+1)*stride] } else { t0, t1 = conj(t0), conj(t1) }
                        w := simd.f32x4{real(t0), imag(t0), real(t1), imag(t1)}
                        vw := simd_cmul2_f32x4(v, w)
                        intrinsics.unaligned_store(cast(^simd.f32x4)(&data[iu]), u+vw)
                        intrinsics.unaligned_store(cast(^simd.f32x4)(&data[iv]), u-vw)
                    }
                }
                #no_bounds_check for ; k < half; k += 1 {
                    u := data[base+k]
                    tw := plan.twiddles[k*stride]
                    if plan.twiddles_inv != nil { tw = plan.twiddles_inv[k*stride] } else { tw = conj(tw) }
                    v := data[base+k+half] * tw
                    data[base+k] = u + v; data[base+k+half] = u - v
                }
            }
            len <<= 1
        }
    }
    inv_n := f32(1.0 / f64(n))
    i = 0
    when FFT_USE_SIMD_KERNELS {
        for ; i+3 < n; i += 4 {
            v := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i]))
            intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i]), simd_scale_complex64x4(v, inv_n))
        }
        for ; i+1 < n; i += 2 {
            v := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i]))
            intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i]), simd_scale_complex64x2(v, inv_n))
        }
    }
    #no_bounds_check for ; i < n; i += 1 { data[i] *= complex(inv_n, f32(0.0)) }
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

c2c_plan_2d_destroy_f32 :: proc(plan: ^C2C_Plan_2D_F32) {
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	if plan.row_plan.initialized {
		c2c_plan_destroy_f32(&plan.row_plan)
	}
	if plan.col_plan.initialized {
		c2c_plan_destroy_f32(&plan.col_plan)
	}
	plan^ = {}
}

c2c_plan_2d_init_f32_with_backend :: proc(plan: ^C2C_Plan_2D_F32, rows, cols: int, backend: Backend, allocator := context.allocator) -> Error {
	if rows < 1 || cols < 1 {
		return .Invalid_Length
	}
	c2c_plan_2d_destroy_f32(plan)

	row_err := c2c_plan_init_f32_with_backend(&plan.row_plan, cols, backend, allocator)
	if row_err != .None {
		return row_err
	}
	col_err := c2c_plan_init_f32_with_backend(&plan.col_plan, rows, backend, allocator)
	if col_err != .None {
		c2c_plan_destroy_f32(&plan.row_plan)
		return col_err
	}

	scratch, scratch_err := make([]complex64, rows, allocator)
	if scratch_err != .None {
		c2c_plan_destroy_f32(&plan.row_plan)
		c2c_plan_destroy_f32(&plan.col_plan)
		return .Allocation_Failed
	}

	plan.rows = rows
	plan.cols = cols
	if plan.row_plan.backend == plan.col_plan.backend {
		plan.backend = plan.row_plan.backend
	} else {
		plan.backend = .Auto
	}
	plan.scratch = scratch
	plan.allocator = allocator
	plan.initialized = true
	return .None
}

c2c_plan_2d_init_f32 :: proc(plan: ^C2C_Plan_2D_F32, rows, cols: int, allocator := context.allocator) -> Error {
	return c2c_plan_2d_init_f32_with_backend(plan, rows, cols, .Auto, allocator)
}

c2c_2d_forward_in_place_f32_with_scratch :: proc(plan: ^C2C_Plan_2D_F32, data: []complex64, scratch: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.rows*plan.cols || len(scratch) != plan.rows {
		return .Size_Mismatch
	}

	for r in 0..<plan.rows {
		row := data[r*plan.cols:][:plan.cols]
		if err := c2c_forward_in_place_f32(&plan.row_plan, row); err != .None {
			return err
		}
	}

	for c in 0..<plan.cols {
		#no_bounds_check for r in 0..<plan.rows {
			scratch[r] = data[r*plan.cols+c]
		}
		if err := c2c_forward_in_place_f32(&plan.col_plan, scratch[:plan.rows]); err != .None {
			return err
		}
		#no_bounds_check for r in 0..<plan.rows {
			data[r*plan.cols+c] = scratch[r]
		}
	}

	return .None
}

c2c_2d_inverse_in_place_f32_with_scratch :: proc(plan: ^C2C_Plan_2D_F32, data: []complex64, scratch: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.rows*plan.cols || len(scratch) != plan.rows {
		return .Size_Mismatch
	}

	for r in 0..<plan.rows {
		row := data[r*plan.cols:][:plan.cols]
		if err := c2c_inverse_in_place_f32(&plan.row_plan, row); err != .None {
			return err
		}
	}

	for c in 0..<plan.cols {
		#no_bounds_check for r in 0..<plan.rows {
			scratch[r] = data[r*plan.cols+c]
		}
		if err := c2c_inverse_in_place_f32(&plan.col_plan, scratch[:plan.rows]); err != .None {
			return err
		}
		#no_bounds_check for r in 0..<plan.rows {
			data[r*plan.cols+c] = scratch[r]
		}
	}

	return .None
}

c2c_2d_forward_in_place_f32 :: proc(plan: ^C2C_Plan_2D_F32, data: []complex64) -> Error {
	return c2c_2d_forward_in_place_f32_with_scratch(plan, data, plan.scratch)
}

c2c_2d_inverse_in_place_f32 :: proc(plan: ^C2C_Plan_2D_F32, data: []complex64) -> Error {
	return c2c_2d_inverse_in_place_f32_with_scratch(plan, data, plan.scratch)
}

c2c_2d_forward_f32 :: proc(plan: ^C2C_Plan_2D_F32, input: []complex64, output: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_2d_forward_in_place_f32(plan, output)
}

c2c_2d_inverse_f32 :: proc(plan: ^C2C_Plan_2D_F32, input: []complex64, output: []complex64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_2d_inverse_in_place_f32(plan, output)
}
complex64_dft4_forward :: #force_inline proc(a, b, c, d: complex64) -> (y0, y1, y2, y3: complex64) {
	t0 := a + c
	t1 := a - c
	t2 := b + d
	bd := b - d
	t3 := complex(imag(bd), -real(bd))
	y0 = t0 + t2
	y1 = t1 + t3
	y2 = t0 - t2
	y3 = t1 - t3
	return
}
complex64_dft4_inverse :: #force_inline proc(a, b, c, d: complex64) -> (y0, y1, y2, y3: complex64) {
	t0 := a + c
	t1 := a - c
	t2 := b + d
	bd := b - d
	t3 := complex(-imag(bd), real(bd))
	y0 = t0 + t2
	y1 = t1 + t3
	y2 = t0 - t2
	y3 = t1 - t3
	return
}
digit_reverse4_permute_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) {
	digit_count := plan.log2_n / 2
	#no_bounds_check for i in 0..<plan.n {
		j := int(reverse_base4_u32(u32(i), digit_count))
		if j > i { data[i], data[j] = data[j], data[i] }
	}
}
simd_radix4_butterfly_f32x4 :: #force_inline proc(a, b, c, d, w1, w2, w3, rotation: simd.f32x4) -> (y0, y1, y2, y3: simd.f32x4) {
    bv := simd_cmul2_f32x4(b, w1)
    cv := simd_cmul2_f32x4(c, w2)
    dv := simd_cmul2_f32x4(d, w3)
    t0 := simd.add(a, cv)
    t1 := simd.sub(a, cv)
    t2 := simd.add(bv, dv)
    bd := simd.sub(bv, dv)
    t3 := simd.mul(simd.shuffle(bd, bd, 1, 0, 3, 2), rotation)
    y0 = simd.add(t0, t2)
    y1 = simd.add(t1, t3)
    y2 = simd.sub(t0, t2)
    y3 = simd.sub(t1, t3)
    return
}
cooley_tukey_radix4_stage_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64, stage_len: int, inverse: bool) {
    quarter := stage_len / 4
    stride := plan.n / stage_len
    half_n := plan.n / 2
    rotation8 := simd.f32x8{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0}
    rotation4 := simd.f32x4{1.0, -1.0, 1.0, -1.0}
    if inverse {
        rotation8 = simd.f32x8{-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0}
        rotation4 = simd.f32x4{-1.0, 1.0, -1.0, 1.0}
    }
    stage_index := (log2_exact(stage_len) - 4) / 2
    packed_stage: ^Radix4_F32_SIMD_Stage
    if plan.radix4_simd_stages != nil && plan.n >= 4096 {
        packed_stage = &plan.radix4_simd_stages[stage_index]
    }
    for base := 0; base < plan.n; base += stage_len {
        k := 0
        when FFT_USE_SIMD_KERNELS {
            if packed_stage != nil {
                p := 0
                conj_sign := simd.f32x8{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0}
                for ; k+3 < quarter; k += 4 {
                    i0 := base + k
                    i1 := i0 + quarter
                    i2 := i1 + quarter
                    i3 := i2 + quarter
                    packed := packed_stage.tw[p]
                    w1 := packed.w1
                    w2 := packed.w2
                    w3 := packed.w3
                    if inverse {
                        w1 = simd.mul(w1, conj_sign)
                        w2 = simd.mul(w2, conj_sign)
                        w3 = simd.mul(w3, conj_sign)
                    }
                    a := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i0]))
                    b := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i1]))
                    c := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i2]))
                    d := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i3]))
                    y0, y1, y2, y3 := simd_radix4_butterfly_f32x8(a, b, c, d, w1, w2, w3, rotation8)
                    intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i0]), y0)
                    intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i1]), y1)
                    intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i2]), y2)
                    intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i3]), y3)
                    p += 1
                }
            }
            for ; k+3 < quarter && plan.n >= 4096 && packed_stage == nil; k += 4 {
                i0 := base + k
                i1 := i0 + quarter
                i2 := i1 + quarter
                i3 := i2 + quarter
                idx1 := k * stride
                idx1b := idx1 + stride
                idx1c := idx1b + stride
                idx1d := idx1c + stride
                idx2 := k * stride * 2
                idx2b := idx2 + stride * 2
                idx2c := idx2b + stride * 2
                idx2d := idx2c + stride * 2
                idx3 := k * stride * 3
                idx3b := idx3 + stride * 3
                idx3c := idx3b + stride * 3
                idx3d := idx3c + stride * 3
                w1a := plan.twiddles[idx1]
                w1b := plan.twiddles[idx1b]
                w1c := plan.twiddles[idx1c]
                w1d := plan.twiddles[idx1d]
                w2a := plan.twiddles[idx2]
                w2b := plan.twiddles[idx2b]
                w2c := plan.twiddles[idx2c]
                w2d := plan.twiddles[idx2d]
                w3a := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
                w3b := plan.twiddles[idx3b] if idx3b < half_n else -plan.twiddles[idx3b-half_n]
                w3c := plan.twiddles[idx3c] if idx3c < half_n else -plan.twiddles[idx3c-half_n]
                w3d := plan.twiddles[idx3d] if idx3d < half_n else -plan.twiddles[idx3d-half_n]
                if inverse {
                    if plan.twiddles_inv != nil {
                        w1a = plan.twiddles_inv[idx1]; w1b = plan.twiddles_inv[idx1b]; w1c = plan.twiddles_inv[idx1c]; w1d = plan.twiddles_inv[idx1d]
                        w2a = plan.twiddles_inv[idx2]; w2b = plan.twiddles_inv[idx2b]; w2c = plan.twiddles_inv[idx2c]; w2d = plan.twiddles_inv[idx2d]
                        w3a = plan.twiddles_inv[idx3] if idx3 < half_n else -plan.twiddles_inv[idx3-half_n]
                        w3b = plan.twiddles_inv[idx3b] if idx3b < half_n else -plan.twiddles_inv[idx3b-half_n]
                        w3c = plan.twiddles_inv[idx3c] if idx3c < half_n else -plan.twiddles_inv[idx3c-half_n]
                        w3d = plan.twiddles_inv[idx3d] if idx3d < half_n else -plan.twiddles_inv[idx3d-half_n]
                    } else {
                        w1a, w1b, w1c, w1d = conj(w1a), conj(w1b), conj(w1c), conj(w1d)
                        w2a, w2b, w2c, w2d = conj(w2a), conj(w2b), conj(w2c), conj(w2d)
                        w3a, w3b, w3c, w3d = conj(w3a), conj(w3b), conj(w3c), conj(w3d)
                    }
                }
                w1 := simd.f32x8{real(w1a), imag(w1a), real(w1b), imag(w1b), real(w1c), imag(w1c), real(w1d), imag(w1d)}
                w2 := simd.f32x8{real(w2a), imag(w2a), real(w2b), imag(w2b), real(w2c), imag(w2c), real(w2d), imag(w2d)}
                w3 := simd.f32x8{real(w3a), imag(w3a), real(w3b), imag(w3b), real(w3c), imag(w3c), real(w3d), imag(w3d)}
                a := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i0]))
                b := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i1]))
                c := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i2]))
                d := intrinsics.unaligned_load(cast(^simd.f32x8)(&data[i3]))
                y0, y1, y2, y3 := simd_radix4_butterfly_f32x8(a, b, c, d, w1, w2, w3, rotation8)
                intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i0]), y0)
                intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i1]), y1)
                intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i2]), y2)
                intrinsics.unaligned_store(cast(^simd.f32x8)(&data[i3]), y3)
            }
            for ; k+1 < quarter; k += 2 {
                i0 := base + k
                i1 := i0 + quarter
                i2 := i1 + quarter
                i3 := i2 + quarter
                idx1 := k * stride
                idx1b := idx1 + stride
                idx2 := k * stride * 2
                idx2b := idx2 + stride * 2
                idx3 := k * stride * 3
                idx3b := idx3 + stride * 3
                w1a := plan.twiddles[idx1]; w1b := plan.twiddles[idx1b]
                w2a := plan.twiddles[idx2]; w2b := plan.twiddles[idx2b]
                w3a := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
                w3b := plan.twiddles[idx3b] if idx3b < half_n else -plan.twiddles[idx3b-half_n]
                if inverse {
                    if plan.twiddles_inv != nil {
                        w1a = plan.twiddles_inv[idx1]; w1b = plan.twiddles_inv[idx1b]
                        w2a = plan.twiddles_inv[idx2]; w2b = plan.twiddles_inv[idx2b]
                        w3a = plan.twiddles_inv[idx3] if idx3 < half_n else -plan.twiddles_inv[idx3-half_n]
                        w3b = plan.twiddles_inv[idx3b] if idx3b < half_n else -plan.twiddles_inv[idx3b-half_n]
                    } else {
                        w1a, w1b = conj(w1a), conj(w1b); w2a, w2b = conj(w2a), conj(w2b); w3a, w3b = conj(w3a), conj(w3b)
                    }
                }
                w1 := simd.f32x4{real(w1a), imag(w1a), real(w1b), imag(w1b)}
                w2 := simd.f32x4{real(w2a), imag(w2a), real(w2b), imag(w2b)}
                w3 := simd.f32x4{real(w3a), imag(w3a), real(w3b), imag(w3b)}
                a := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i0]))
                b := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i1]))
                c := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i2]))
                d := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i3]))
                y0, y1, y2, y3 := simd_radix4_butterfly_f32x4(a, b, c, d, w1, w2, w3, rotation4)
                intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i0]), y0)
                intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i1]), y1)
                intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i2]), y2)
                intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i3]), y3)
            }
        }
        #no_bounds_check for ; k < quarter; k += 1 {
            i0 := base + k; i1 := i0 + quarter; i2 := i1 + quarter; i3 := i2 + quarter
            idx1 := k * stride; idx2 := k * stride * 2; idx3 := k * stride * 3
            w1 := plan.twiddles[idx1]; w2 := plan.twiddles[idx2]
            w3 := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
            if inverse {
                if plan.twiddles_inv != nil {
                    w1 = plan.twiddles_inv[idx1]; w2 = plan.twiddles_inv[idx2]
                    w3 = plan.twiddles_inv[idx3] if idx3 < half_n else -plan.twiddles_inv[idx3-half_n]
                } else { w1, w2, w3 = conj(w1), conj(w2), conj(w3) }
            }
            a := data[i0]; b := data[i1]*w1; c := data[i2]*w2; d := data[i3]*w3
            t0 := a+c; t1 := a-c; t2 := b+d; bd := b-d
            t3 := complex(imag(bd), -real(bd)); if inverse { t3 = complex(-imag(bd), real(bd)) }
            data[i0], data[i1], data[i2], data[i3] = t0+t2, t1+t3, t0-t2, t1-t3
        }
    }
}
cooley_tukey_forward_radix4_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) -> Error {
    digit_reverse4_permute_in_place_f32(plan, data)
    for base := 0; base < plan.n; base += 4 {
        data[base], data[base+1], data[base+2], data[base+3] = complex64_dft4_forward(data[base], data[base+1], data[base+2], data[base+3])
    }
    for stage_len := 16; stage_len <= plan.n; stage_len *= 4 {
        cooley_tukey_radix4_stage_f32(plan, data, stage_len, false)
    }
    return .None
}
cooley_tukey_inverse_radix4_in_place_f32 :: proc(plan: ^C2C_Plan_F32, data: []complex64) -> Error {
    digit_reverse4_permute_in_place_f32(plan, data)
    for base := 0; base < plan.n; base += 4 {
        data[base], data[base+1], data[base+2], data[base+3] = complex64_dft4_inverse(data[base], data[base+1], data[base+2], data[base+3])
    }
    for stage_len := 16; stage_len <= plan.n; stage_len *= 4 {
        cooley_tukey_radix4_stage_f32(plan, data, stage_len, true)
    }
    inv_n := f32(1.0 / f64(plan.n))
    i := 0
    when FFT_USE_SIMD_KERNELS {
        for ; i+1 < plan.n; i += 2 {
            v := intrinsics.unaligned_load(cast(^simd.f32x4)(&data[i]))
            intrinsics.unaligned_store(cast(^simd.f32x4)(&data[i]), simd_scale_complex64x2(v, inv_n))
        }
    }
    for ; i < plan.n; i += 1 { data[i] *= complex(inv_n, 0) }
    return .None
}
