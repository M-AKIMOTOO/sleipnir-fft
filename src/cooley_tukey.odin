package fft

import "base:intrinsics"
import "core:simd"
import "core:sync"
import "core:thread"

COOLEY_TUKEY_PARALLEL_MIN_N :: 1 << 16
COOLEY_TUKEY_PARALLEL_MIN_BLOCKS_PER_WORKER :: 16
RADIX4_PARALLEL_CHUNK_MIN_LEN :: 1 << 20
// SIMD feature gates and signs are defined in fft.odin.

COOLEY_W8_FWD_1 :: complex(0.7071067811865475244, -0.7071067811865475244)
COOLEY_W8_FWD_2 :: complex(0.0, -1.0)
COOLEY_W8_FWD_3 :: complex(-0.7071067811865475244, -0.7071067811865475244)
COOLEY_W8_INV_1 :: complex(0.7071067811865475244, 0.7071067811865475244)
COOLEY_W8_INV_2 :: complex(0.0, 1.0)
COOLEY_W8_INV_3 :: complex(-0.7071067811865475244, 0.7071067811865475244)

simd_cmul2_f64x4 :: #force_inline proc(a, b: simd.f64x4) -> simd.f64x4 {
	when intrinsics.has_target_feature("fma") {
		ar := simd.shuffle(a, a, 0, 0, 2, 2)
		ai := simd.shuffle(a, a, 1, 1, 3, 3)
		br := simd.shuffle(b, b, 0, 0, 2, 2)
		bi := simd.shuffle(b, b, 1, 1, 3, 3)
		// Keep one product per component as the FMA addend. This reduces the
		// complex multiply from six vector arithmetic instructions to four.
		re := simd.fused_mul_add(simd.neg(ai), bi, simd.mul(ar, br))
		im := simd.fused_mul_add(ai, br, simd.mul(ar, bi))
		return simd.shuffle(re, im, 0, 4, 2, 6)
	} else {
		ar := simd.shuffle(a, a, 0, 0, 2, 2)
		ai := simd.shuffle(a, a, 1, 1, 3, 3)
		br := simd.shuffle(b, b, 0, 0, 2, 2)
		bi := simd.shuffle(b, b, 1, 1, 3, 3)
		re := simd.sub(simd.mul(ar, br), simd.mul(ai, bi))
		im := simd.add(simd.mul(ar, bi), simd.mul(ai, br))
		return simd.shuffle(re, im, 0, 4, 2, 6)
	}
}

simd_rot_fwd_bd :: #force_inline proc(bd: simd.f64x4) -> simd.f64x4 {
	return simd.mul(simd.shuffle(bd, bd, 1, 0, 3, 2), SIMD_FWD_T3_SIGN)
}

simd_rot_inv_bd :: #force_inline proc(bd: simd.f64x4) -> simd.f64x4 {
	return simd.mul(simd.shuffle(bd, bd, 1, 0, 3, 2), SIMD_INV_T3_SIGN)
}


cooley_tukey_dft4_from_bitrev_forward :: #force_inline proc(a0, a1, a2, a3: complex128) -> (y0, y1, y2, y3: complex128) {
	s0 := a0 + a1
	d0 := a0 - a1
	s1 := a2 + a3
	d1 := a2 - a3
	rot := complex(imag(d1), -real(d1))
	y0 = s0 + s1
	y1 = d0 + rot
	y2 = s0 - s1
	y3 = d0 - rot
	return
}

cooley_tukey_dft4_from_bitrev_inverse :: #force_inline proc(a0, a1, a2, a3: complex128) -> (y0, y1, y2, y3: complex128) {
	s0 := a0 + a1
	d0 := a0 - a1
	s1 := a2 + a3
	d1 := a2 - a3
	rot := complex(-imag(d1), real(d1))
	y0 = s0 + s1
	y1 = d0 + rot
	y2 = s0 - s1
	y3 = d0 - rot
	return
}

cooley_tukey_inverse_radix4_base_block :: #force_inline proc(data: []complex128, base: int) {
	a := data[base+0]
	b := data[base+1]
	c := data[base+2]
	d := data[base+3]

	t0 := a + c
	t1 := a - c
	t2 := b + d
	bd := b - d
	t3 := complex(-imag(bd), real(bd))

	data[base+0] = t0 + t2
	data[base+1] = t1 + t3
	data[base+2] = t0 - t2
	data[base+3] = t1 - t3
}

cooley_tukey_dft8_from_bitrev_forward :: #force_inline proc(
	a0, a1, a2, a3, a4, a5, a6, a7: complex128,
) -> (
	y0, y1, y2, y3, y4, y5, y6, y7: complex128,
) {
	e0, e1, e2, e3 := cooley_tukey_dft4_from_bitrev_forward(a0, a1, a2, a3)
	o0, o1, o2, o3 := cooley_tukey_dft4_from_bitrev_forward(a4, a5, a6, a7)
	t1 := COOLEY_W8_FWD_1 * o1
	t2 := COOLEY_W8_FWD_2 * o2
	t3 := COOLEY_W8_FWD_3 * o3
	y0 = e0 + o0
	y4 = e0 - o0
	y1 = e1 + t1
	y5 = e1 - t1
	y2 = e2 + t2
	y6 = e2 - t2
	y3 = e3 + t3
	y7 = e3 - t3
	return
}

cooley_tukey_dft8_from_bitrev_inverse :: #force_inline proc(
	a0, a1, a2, a3, a4, a5, a6, a7: complex128,
) -> (
	y0, y1, y2, y3, y4, y5, y6, y7: complex128,
) {
	e0, e1, e2, e3 := cooley_tukey_dft4_from_bitrev_inverse(a0, a1, a2, a3)
	o0, o1, o2, o3 := cooley_tukey_dft4_from_bitrev_inverse(a4, a5, a6, a7)
	t1 := COOLEY_W8_INV_1 * o1
	t2 := COOLEY_W8_INV_2 * o2
	t3 := COOLEY_W8_INV_3 * o3
	y0 = e0 + o0
	y4 = e0 - o0
	y1 = e1 + t1
	y5 = e1 - t1
	y2 = e2 + t2
	y6 = e2 - t2
	y3 = e3 + t3
	y7 = e3 - t3
	return
}

cooley_tukey_forward_radix8_base_blocks :: proc(data: []complex128, block_start, block_end: int) {
	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * 8
		data[base+0], data[base+1], data[base+2], data[base+3],
		data[base+4], data[base+5], data[base+6], data[base+7] =
			cooley_tukey_dft8_from_bitrev_forward(
				data[base+0], data[base+1], data[base+2], data[base+3],
				data[base+4], data[base+5], data[base+6], data[base+7],
			)
	}
}

cooley_tukey_inverse_radix8_base_blocks :: proc(data: []complex128, block_start, block_end: int) {
	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * 8
		data[base+0], data[base+1], data[base+2], data[base+3],
		data[base+4], data[base+5], data[base+6], data[base+7] =
			cooley_tukey_dft8_from_bitrev_inverse(
				data[base+0], data[base+1], data[base+2], data[base+3],
				data[base+4], data[base+5], data[base+6], data[base+7],
			)
	}
}

reverse_mixed_radix8_base4_u32 :: #force_inline proc(x: u32, radix4_digit_count: int) -> u32 {
	r := x & 0b111
	v := x >> 3
	#no_bounds_check for _ in 0..<radix4_digit_count {
		r = (r << 2) | (v & 0b11)
		v >>= 2
	}
	return r
}

cooley_tukey_digit_reverse8_permute_in_place :: proc(plan: ^C2C_Plan, data: []complex128) {
	n := plan.n
	radix4_digit_count := (plan.log2_n - 3) / 2
	#no_bounds_check for i in 0..<n {
		j := int(reverse_mixed_radix8_base4_u32(u32(i), radix4_digit_count))
		if j > i {
			data[i], data[j] = data[j], data[i]
		}
	}
}

cooley_tukey_find_radix2_stage :: #force_inline proc(plan: ^C2C_Plan, stage_len: int) -> ^Radix2_SIMD_Stage {
	if plan.radix2_simd_stages == nil {
		return nil
	}
	for i in 0..<len(plan.radix2_simd_stages) {
		if plan.radix2_simd_stages[i].len == stage_len {
			return &plan.radix2_simd_stages[i]
		}
	}
	return nil
}

cooley_tukey_radix2_simd_destroy :: proc(plan: ^C2C_Plan) {
	if plan.radix2_simd_stages == nil {
		return
	}
	for i in 0..<len(plan.radix2_simd_stages) {
		if plan.radix2_simd_stages[i].tw != nil {
			delete(plan.radix2_simd_stages[i].tw, plan.allocator)
		}
		if plan.radix2_simd_stages[i].tw_inv != nil {
			delete(plan.radix2_simd_stages[i].tw_inv, plan.allocator)
		}
	}
	delete(plan.radix2_simd_stages, plan.allocator)
	plan.radix2_simd_stages = nil
}

cooley_tukey_radix2_simd_init :: proc(plan: ^C2C_Plan) -> Error {
	if plan.cooley_radix != 2 || plan.n < 8 {
		return .None
	}
	if plan.radix2_simd_stages != nil {
		return .None
	}
	when !FFT_USE_SIMD_KERNELS {
		return .None
	}

	stage_count := 0
	for stage_len := 4; stage_len <= plan.n; stage_len *= 2 {
		stage_count += 1
	}
	stages, stages_err := make([]Radix2_SIMD_Stage, stage_count, plan.allocator)
	if stages_err != .None {
		return .Allocation_Failed
	}

	tw := plan.twiddles
	stage_idx := 0
	for stage_len := 4; stage_len <= plan.n; stage_len *= 2 {
		half := stage_len / 2
		stride := plan.n / stage_len
		pair_count := (half - 1) / 2
		stages[stage_idx].len = stage_len
		if pair_count > 0 {
			tw_pairs, tw_err := make([]Radix2_Twiddle_Pair_Pack, pair_count, plan.allocator)
			if tw_err != .None {
				for j in 0..<stage_idx {
					if stages[j].tw != nil do delete(stages[j].tw, plan.allocator)
				}
				delete(stages, plan.allocator)
				return .Allocation_Failed
			}

			for p in 0..<pair_count {
				k := 1 + 2*p
				idx := k * stride
				idxn := idx + stride
				t0 := tw[idx]
				t1 := tw[idxn]
				tw_pairs[p] = Radix2_Twiddle_Pair_Pack{
					w0_re = real(t0), w0_im = imag(t0),
					w1_re = real(t1), w1_im = imag(t1),
				}
			}
			stages[stage_idx].tw = tw_pairs
		}
		stage_idx += 1
	}

	plan.radix2_simd_stages = stages
	return .None
}

cooley_tukey_find_radix4_stage :: #force_inline proc(plan: ^C2C_Plan, stage_len: int) -> ^Radix4_SIMD_Stage {
	if plan.radix4_simd_stages == nil {
		return nil
	}
	for i in 0..<len(plan.radix4_simd_stages) {
		if plan.radix4_simd_stages[i].len == stage_len {
			return &plan.radix4_simd_stages[i]
		}
	}
	return nil
}

cooley_tukey_radix4_simd_destroy :: proc(plan: ^C2C_Plan) {
	if plan.radix4_simd_stages == nil {
		return
	}
	for i in 0..<len(plan.radix4_simd_stages) {
		if plan.radix4_simd_stages[i].tw != nil {
			delete(plan.radix4_simd_stages[i].tw, plan.allocator)
		}
	}
	delete(plan.radix4_simd_stages, plan.allocator)
	plan.radix4_simd_stages = nil
}

cooley_tukey_radix4_simd_init :: proc(plan: ^C2C_Plan) -> Error {
	if plan.cooley_radix != 4 || plan.n < 16 {
		return .None
	}
	if plan.radix4_simd_stages != nil {
		return .None
	}
	when !FFT_USE_SIMD_KERNELS {
		return .None
	}

	stage_count := 0
	for len := 4; len <= plan.n; len *= 4 {
		stage_count += 1
	}
	stages, stages_err := make([]Radix4_SIMD_Stage, stage_count, plan.allocator)
	if stages_err != .None {
		return .Allocation_Failed
	}

	tw := plan.twiddles
	half_n := plan.n / 2
	stage_idx := 0
	for len := 4; len <= plan.n; len *= 4 {
		quarter := len / 4
		stride := plan.n / len
		stride2 := stride * 2
		stride3 := stride * 3
		stages[stage_idx].len = len
		pair_count := quarter / 2
		if pair_count > 0 {
			tw_packed, tw_err := make([]Radix4_Twiddle_Triple_Pack, pair_count, plan.allocator)
			if tw_err != .None {
				if tw_packed != nil do delete(tw_packed, plan.allocator)
				for j in 0..<stage_idx {
					if stages[j].tw != nil do delete(stages[j].tw, plan.allocator)
				}
				delete(stages, plan.allocator)
				return .Allocation_Failed
			}

			for p in 0..<pair_count {
				k := p * 2
				idx1 := k * stride
				idx2 := k * stride2
				idx3 := k * stride3
				idx1n := idx1 + stride
				idx2n := idx2 + stride2
				idx3n := idx3 + stride3
				t3a := tw[idx3] if idx3 < half_n else -tw[idx3-half_n]
				t3b := tw[idx3n] if idx3n < half_n else -tw[idx3n-half_n]
				tw_packed[p] = Radix4_Twiddle_Triple_Pack{
					w1 = Radix4_Twiddle_Pair_Pack{
						w0_re = real(tw[idx1]),  w0_im = imag(tw[idx1]),
						w1_re = real(tw[idx1n]), w1_im = imag(tw[idx1n]),
					},
					w2 = Radix4_Twiddle_Pair_Pack{
						w0_re = real(tw[idx2]),  w0_im = imag(tw[idx2]),
						w1_re = real(tw[idx2n]), w1_im = imag(tw[idx2n]),
					},
					w3 = Radix4_Twiddle_Pair_Pack{
						w0_re = real(t3a), w0_im = imag(t3a),
						w1_re = real(t3b), w1_im = imag(t3b),
					},
				}
			}
			stages[stage_idx].tw = tw_packed
		}
		stage_idx += 1
	}

	plan.radix4_simd_stages = stages
	return .None
}

cooley_tukey_forward_stage_blocks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, half, stride: int,
	block_start, block_end: int,
) {
	tw := plan.twiddles
	stage := cooley_tukey_find_radix2_stage(plan, len)
	if stage != nil && stage.tw != nil {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * len
			u0 := data[base]
			v0 := data[base+half]
			data[base] = u0 + v0
			data[base+half] = u0 - v0
			k := 1
			idx := stride
			pair_count := (half - 1) / 2
			for p in 0..<pair_count {
				iu := base + k
				iv := iu + half


				u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
				v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

				w := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p]))
				vw := simd_cmul2_f64x4(v, w)
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.add(u, vw))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.sub(u, vw))

				k += 2
				idx += 2 * stride
			}
			for ; k < half; k += 1 {
				u := data[base+k]
				v := data[base+k+half] * tw[idx]
				data[base+k] = u + v
				data[base+k+half] = u - v
				idx += stride
			}
		}
		return
	}

	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * len
		u0 := data[base]
		v0 := data[base+half]
		data[base] = u0 + v0
		data[base+half] = u0 - v0
		k := 1
		idx := stride
		when FFT_USE_SIMD_KERNELS {
			for k+1 < half {
				iu := base + k
				iv := iu + half

				u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
				v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

				idxn := idx + stride
				w := simd.f64x4{
					real(tw[idx]), imag(tw[idx]),
					real(tw[idxn]), imag(tw[idxn]),
				}
				vw := simd_cmul2_f64x4(v, w)
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.add(u, vw))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.sub(u, vw))

				k += 2
				idx += 2 * stride
			}
		}
		for ; k < half; k += 1 {
			u := data[base+k]
			v := data[base+k+half] * tw[idx]
			data[base+k] = u + v
			data[base+k+half] = u - v
			idx += stride
		}
	}
}

cooley_tukey_inverse_stage_inv_blocks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, half, stride: int,
	block_start, block_end: int,
) {
	tw_inv := plan.twiddles_inv
	stage := cooley_tukey_find_radix2_stage(plan, len)
	if stage != nil && stage.tw != nil {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * len
			u0 := data[base]
			v0 := data[base+half]
			data[base] = u0 + v0
			data[base+half] = u0 - v0
			k := 1
			idx := stride
			pair_count := (half - 1) / 2
			for p in 0..<pair_count {
				iu := base + k
				iv := iu + half


				u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
				v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

				w := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p])), SIMD_CONJ_SIGN)
				vw := simd_cmul2_f64x4(v, w)
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.add(u, vw))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.sub(u, vw))

				k += 2
				idx += 2 * stride
			}
			for ; k < half; k += 1 {
				u := data[base+k]
				v := data[base+k+half] * tw_inv[idx]
				data[base+k] = u + v
				data[base+k+half] = u - v
				idx += stride
			}
		}
		return
	}

	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * len
		u0 := data[base]
		v0 := data[base+half]
		data[base] = u0 + v0
		data[base+half] = u0 - v0
		k := 1
		idx := stride
		when FFT_USE_SIMD_KERNELS {
			for k+1 < half {
				iu := base + k
				iv := iu + half

				u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
				v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

				idxn := idx + stride
				w := simd.f64x4{
					real(tw_inv[idx]), imag(tw_inv[idx]),
					real(tw_inv[idxn]), imag(tw_inv[idxn]),
				}
				vw := simd_cmul2_f64x4(v, w)
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.add(u, vw))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.sub(u, vw))

				k += 2
				idx += 2 * stride
			}
		}
		for ; k < half; k += 1 {
			u := data[base+k]
			v := data[base+k+half] * tw_inv[idx]
			data[base+k] = u + v
			data[base+k+half] = u - v
			idx += stride
		}
	}
}

cooley_tukey_inverse_stage_conj_blocks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, half, stride: int,
	block_start, block_end: int,
) {
	tw := plan.twiddles
	stage := cooley_tukey_find_radix2_stage(plan, len)
	if stage != nil && stage.tw != nil {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * len
			u0 := data[base]
			v0 := data[base+half]
			data[base] = u0 + v0
			data[base+half] = u0 - v0
			k := 1
			idx := stride
			pair_count := (half - 1) / 2
			for p in 0..<pair_count {
				iu := base + k
				iv := iu + half


				u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
				v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

				w := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p])), SIMD_CONJ_SIGN)
				vw := simd_cmul2_f64x4(v, w)
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.add(u, vw))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.sub(u, vw))

				k += 2
				idx += 2 * stride
			}
			for ; k < half; k += 1 {
				u := data[base+k]
				t := tw[idx]
				v := data[base+k+half] * complex(real(t), -imag(t))
				data[base+k] = u + v
				data[base+k+half] = u - v
				idx += stride
			}
		}
		return
	}

	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * len
		u0 := data[base]
		v0 := data[base+half]
		data[base] = u0 + v0
		data[base+half] = u0 - v0
		k := 1
		idx := stride
		when FFT_USE_SIMD_KERNELS {
			for k+1 < half {
				iu := base + k
				iv := iu + half

				u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
				v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

				idxn := idx + stride
				w := simd.f64x4{
					real(tw[idx]), -imag(tw[idx]),
					real(tw[idxn]), -imag(tw[idxn]),
				}
				vw := simd_cmul2_f64x4(v, w)
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.add(u, vw))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.sub(u, vw))

				k += 2
				idx += 2 * stride
			}
		}
		for ; k < half; k += 1 {
			u := data[base+k]
			t := tw[idx]
			v := data[base+k+half] * complex(real(t), -imag(t))
			data[base+k] = u + v
			data[base+k+half] = u - v
			idx += stride
		}
	}
}

cooley_tukey_inverse_stage_scaled :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, half, stride: int,
	scale: f64,
	use_inv: bool,
) {
	stage := cooley_tukey_find_radix2_stage(plan, len)
	scale_c := complex(scale, 0.0)
	scale_vec := simd.f64x4{scale, scale, scale, scale}
	if stage != nil && stage.tw != nil {
		base := 0
		u0 := data[base]
		v0 := data[base+half]
		data[base] = (u0 + v0) * scale_c
		data[base+half] = (u0 - v0) * scale_c
		k := 1
		pair_count := (half - 1) / 2
		for p in 0..<pair_count {
			iu := base + k
			iv := iu + half

			u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
			v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))
			w := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p])), SIMD_CONJ_SIGN)
			vw := simd_cmul2_f64x4(v, w)
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.mul(simd.add(u, vw), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.mul(simd.sub(u, vw), scale_vec))

			k += 2
		}
		idx := k * stride
		for ; k < half; k += 1 {
			u := data[base+k]
			if use_inv {
				v := data[base+k+half] * plan.twiddles_inv[idx]
				data[base+k] = (u + v) * scale_c
				data[base+k+half] = (u - v) * scale_c
			} else {
				t := plan.twiddles[idx]
				v := data[base+k+half] * complex(real(t), -imag(t))
				data[base+k] = (u + v) * scale_c
				data[base+k+half] = (u - v) * scale_c
			}
			idx += stride
		}
		return
	}

	base := 0
	u0 := data[base]
	v0 := data[base+half]
	data[base] = (u0 + v0) * scale_c
	data[base+half] = (u0 - v0) * scale_c
	k := 1
	idx := stride
	when FFT_USE_SIMD_KERNELS {
		for k+1 < half {
			iu := base + k
			iv := iu + half

			u := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iu]))
			v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[iv]))

			idxn := idx + stride
			w: simd.f64x4
			if use_inv {
				w = simd.f64x4{
					real(plan.twiddles_inv[idx]), imag(plan.twiddles_inv[idx]),
					real(plan.twiddles_inv[idxn]), imag(plan.twiddles_inv[idxn]),
				}
			} else {
				t0 := plan.twiddles[idx]
				t1 := plan.twiddles[idxn]
				w = simd.f64x4{
					real(t0), -imag(t0),
					real(t1), -imag(t1),
				}
			}
			vw := simd_cmul2_f64x4(v, w)
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iu]), simd.mul(simd.add(u, vw), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[iv]), simd.mul(simd.sub(u, vw), scale_vec))

			k += 2
			idx += 2 * stride
		}
	}
	for ; k < half; k += 1 {
		u := data[base+k]
		if use_inv {
			v := data[base+k+half] * plan.twiddles_inv[idx]
			data[base+k] = (u + v) * scale_c
			data[base+k+half] = (u - v) * scale_c
		} else {
			t := plan.twiddles[idx]
			v := data[base+k+half] * complex(real(t), -imag(t))
			data[base+k] = (u + v) * scale_c
			data[base+k+half] = (u - v) * scale_c
		}
		idx += stride
	}
}

cooley_tukey_forward_radix4_stage_blocks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride: int,
	block_start, block_end: int,
) {
	if len == 4 {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * 4
			a := data[base+0]
			b := data[base+1]
			c := data[base+2]
			d := data[base+3]

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(imag(bd), -real(bd))

			data[base+0] = t0 + t2
			data[base+1] = t1 + t3
			data[base+2] = t0 - t2
			data[base+3] = t1 - t3
		}
		return
	}

	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	tw := plan.twiddles
	stage := cooley_tukey_find_radix4_stage(plan, len)

	if stage != nil && stage.tw != nil {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * len
			k := 0
			pair_count := quarter / 2
			for p in 0..<pair_count {
				k = 2 * p
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter


				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))

				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				w1 := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1))
				w2 := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2))
				w3 := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3))

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_fwd_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))
			}

			k = 2 * pair_count
			idx1 := k * stride
			idx2 := (2 * pair_count) * stride2
			idx3 := (2 * pair_count) * stride3
			for ; k < quarter; k += 1 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter

				a := data[i0]
				w1 := tw[idx1]
				w2 := tw[idx2]
				w3 := tw[idx3] if idx3 < half_n else -tw[idx3-half_n]
				b := data[i1] * w1
				c := data[i2] * w2
				d := data[i3] * w3

				t0 := a + c
				t1 := a - c
				t2 := b + d
				bd := b - d
				t3 := complex(imag(bd), -real(bd))

				data[i0] = t0 + t2
				data[i1] = t1 + t3
				data[i2] = t0 - t2
				data[i3] = t1 - t3

				idx1 += stride
				idx2 += stride2
				idx3 += stride3
			}
		}
		return
	}

	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * len
		idx1 := 0
		idx2 := 0
		idx3 := 0
		k := 0
		when FFT_USE_SIMD_KERNELS {
			for k+1 < quarter {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter


				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				idx1n := idx1 + stride
				idx2n := idx2 + stride2
				idx3n := idx3 + stride3
				w1 := simd.f64x4{
					real(tw[idx1]), imag(tw[idx1]),
					real(tw[idx1n]), imag(tw[idx1n]),
				}
				w2 := simd.f64x4{
					real(tw[idx2]), imag(tw[idx2]),
					real(tw[idx2n]), imag(tw[idx2n]),
				}
				tw3a := tw[idx3] if idx3 < half_n else -tw[idx3-half_n]
				tw3b := tw[idx3n] if idx3n < half_n else -tw[idx3n-half_n]
				w3 := simd.f64x4{
					real(tw3a), imag(tw3a),
					real(tw3b), imag(tw3b),
				}

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_fwd_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))

				k += 2
				idx1 += 2 * stride
				idx2 += 2 * stride2
				idx3 += 2 * stride3
			}
		}
		for ; k < quarter; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter

			a := data[i0]
			w1 := tw[idx1]
			w2 := tw[idx2]
			w3 := tw[idx3] if idx3 < half_n else -tw[idx3-half_n]
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(imag(bd), -real(bd))

			data[i0] = t0 + t2
			data[i1] = t1 + t3
			data[i2] = t0 - t2
			data[i3] = t1 - t3

			idx1 += stride
			idx2 += stride2
			idx3 += stride3
		}
	}
}

cooley_tukey_inverse_radix4_stage_inv_blocks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride: int,
	block_start, block_end: int,
) {
	if len == 4 {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			cooley_tukey_inverse_radix4_base_block(data, block * 4)
		}
		return
	}

	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	tw_inv := plan.twiddles_inv
	stage := cooley_tukey_find_radix4_stage(plan, len)
	if stage != nil && stage.tw != nil {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * len
			k := 0
			pair_count := quarter / 2
			for p in 0..<pair_count {
				k = 2 * p
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter


				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))

				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				w1 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1)), SIMD_CONJ_SIGN)
				w2 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2)), SIMD_CONJ_SIGN)
				w3 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3)), SIMD_CONJ_SIGN)

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_inv_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))
			}

			k = 2 * pair_count
			idx1 := k * stride
			idx2 := (2 * pair_count) * stride2
			idx3 := (2 * pair_count) * stride3
			for ; k < quarter; k += 1 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter

				a := data[i0]
				w1 := tw_inv[idx1]
				w2 := tw_inv[idx2]
				w3 := tw_inv[idx3] if idx3 < half_n else -tw_inv[idx3-half_n]
				b := data[i1] * w1
				c := data[i2] * w2
				d := data[i3] * w3

				t0 := a + c
				t1 := a - c
				t2 := b + d
				bd := b - d
				t3 := complex(-imag(bd), real(bd))

				data[i0] = t0 + t2
				data[i1] = t1 + t3
				data[i2] = t0 - t2
				data[i3] = t1 - t3

				idx1 += stride
				idx2 += stride2
				idx3 += stride3
			}
		}
		return
	}

	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * len
		idx1 := 0
		idx2 := 0
		idx3 := 0
		k := 0
		when FFT_USE_SIMD_KERNELS {
			for k+1 < quarter {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter


				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				idx1n := idx1 + stride
				idx2n := idx2 + stride2
				idx3n := idx3 + stride3
				w1 := simd.f64x4{
					real(tw_inv[idx1]), imag(tw_inv[idx1]),
					real(tw_inv[idx1n]), imag(tw_inv[idx1n]),
				}
				w2 := simd.f64x4{
					real(tw_inv[idx2]), imag(tw_inv[idx2]),
					real(tw_inv[idx2n]), imag(tw_inv[idx2n]),
				}
				tw3a := tw_inv[idx3] if idx3 < half_n else -tw_inv[idx3-half_n]
				tw3b := tw_inv[idx3n] if idx3n < half_n else -tw_inv[idx3n-half_n]
				w3 := simd.f64x4{
					real(tw3a), imag(tw3a),
					real(tw3b), imag(tw3b),
				}

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_inv_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))

				k += 2
				idx1 += 2 * stride
				idx2 += 2 * stride2
				idx3 += 2 * stride3
			}
		}
		for ; k < quarter; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter

			a := data[i0]
			w1 := tw_inv[idx1]
			w2 := tw_inv[idx2]
			w3 := tw_inv[idx3] if idx3 < half_n else -tw_inv[idx3-half_n]
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(-imag(bd), real(bd))

			data[i0] = t0 + t2
			data[i1] = t1 + t3
			data[i2] = t0 - t2
			data[i3] = t1 - t3

			idx1 += stride
			idx2 += stride2
			idx3 += stride3
		}
	}
}

cooley_tukey_inverse_radix4_stage_conj_blocks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride: int,
	block_start, block_end: int,
) {
	if len == 4 {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			cooley_tukey_inverse_radix4_base_block(data, block * 4)
		}
		return
	}

	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	tw := plan.twiddles
	stage := cooley_tukey_find_radix4_stage(plan, len)
	if stage != nil && stage.tw != nil {
		#no_bounds_check for block := block_start; block < block_end; block += 1 {
			base := block * len
			k := 0
			pair_count := quarter / 2
			for p in 0..<pair_count {
				k = 2 * p
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter


				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))

				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				w1 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1)), SIMD_CONJ_SIGN)
				w2 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2)), SIMD_CONJ_SIGN)
				w3 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3)), SIMD_CONJ_SIGN)

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_inv_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))
			}

			k = 2 * pair_count
			idx1 := k * stride
			idx2 := (2 * pair_count) * stride2
			idx3 := (2 * pair_count) * stride3
			for ; k < quarter; k += 1 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter

				a := data[i0]
				w1 := conj(tw[idx1])
				w2 := conj(tw[idx2])
				w3 := conj(tw[idx3]) if idx3 < half_n else -conj(tw[idx3-half_n])
				b := data[i1] * w1
				c := data[i2] * w2
				d := data[i3] * w3

				t0 := a + c
				t1 := a - c
				t2 := b + d
				bd := b - d
				t3 := complex(-imag(bd), real(bd))

				data[i0] = t0 + t2
				data[i1] = t1 + t3
				data[i2] = t0 - t2
				data[i3] = t1 - t3

				idx1 += stride
				idx2 += stride2
				idx3 += stride3
			}
		}
		return
	}

	#no_bounds_check for block := block_start; block < block_end; block += 1 {
		base := block * len
		idx1 := 0
		idx2 := 0
		idx3 := 0
		k := 0
		when FFT_USE_SIMD_KERNELS {
			for k+1 < quarter {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter


				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				idx1n := idx1 + stride
				idx2n := idx2 + stride2
				idx3n := idx3 + stride3
				w1a := conj(tw[idx1])
				w1b := conj(tw[idx1n])
				w2a := conj(tw[idx2])
				w2b := conj(tw[idx2n])
				tw3a := conj(tw[idx3]) if idx3 < half_n else -conj(tw[idx3-half_n])
				tw3b := conj(tw[idx3n]) if idx3n < half_n else -conj(tw[idx3n-half_n])
				w1 := simd.f64x4{real(w1a), imag(w1a), real(w1b), imag(w1b)}
				w2 := simd.f64x4{real(w2a), imag(w2a), real(w2b), imag(w2b)}
				w3 := simd.f64x4{real(tw3a), imag(tw3a), real(tw3b), imag(tw3b)}

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_inv_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))

				k += 2
				idx1 += 2 * stride
				idx2 += 2 * stride2
				idx3 += 2 * stride3
			}
		}
		for ; k < quarter; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter

			a := data[i0]
			w1 := conj(tw[idx1])
			w2 := conj(tw[idx2])
			w3 := conj(tw[idx3]) if idx3 < half_n else -conj(tw[idx3-half_n])
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(-imag(bd), real(bd))

			data[i0] = t0 + t2
			data[i1] = t1 + t3
			data[i2] = t0 - t2
			data[i3] = t1 - t3

			idx1 += stride
			idx2 += stride2
			idx3 += stride3
		}
	}
}

cooley_tukey_inverse_radix4_stage_scaled :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride: int,
	scale: f64,
	use_inv: bool,
) {
	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	stage := cooley_tukey_find_radix4_stage(plan, len)
	scale_c := complex(scale, 0.0)
	scale_vec := simd.f64x4{scale, scale, scale, scale}

	if stage != nil && stage.tw != nil {
		base := 0
		k := 0
		pair_count := quarter / 2
		for p in 0..<pair_count {
			k = 2 * p
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter


			a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
			bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
			cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
			dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

			w1 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1)), SIMD_CONJ_SIGN)
			w2 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2)), SIMD_CONJ_SIGN)
			w3 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3)), SIMD_CONJ_SIGN)

			b := simd_cmul2_f64x4(bv, w1)
			c := simd_cmul2_f64x4(cv, w2)
			d := simd_cmul2_f64x4(dv, w3)

			t0 := simd.add(a, c)
			t1 := simd.sub(a, c)
			t2 := simd.add(b, d)
			bd := simd.sub(b, d)
			t3 := simd_rot_inv_bd(bd)

			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.mul(simd.add(t0, t2), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.mul(simd.add(t1, t3), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.mul(simd.sub(t0, t2), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.mul(simd.sub(t1, t3), scale_vec))
		}

		k = 2 * pair_count
		idx1 := k * stride
		idx2 := (2 * pair_count) * stride2
		idx3 := (2 * pair_count) * stride3
		for ; k < quarter; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter

			a := data[i0]
			w1, w2, w3: complex128
			if use_inv {
				w1 = plan.twiddles_inv[idx1]
				w2 = plan.twiddles_inv[idx2]
				w3 = plan.twiddles_inv[idx3] if idx3 < half_n else -plan.twiddles_inv[idx3-half_n]
			} else {
				t1w := plan.twiddles[idx1]
				t2w := plan.twiddles[idx2]
				t3w := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
				w1 = complex(real(t1w), -imag(t1w))
				w2 = complex(real(t2w), -imag(t2w))
				w3 = complex(real(t3w), -imag(t3w))
			}
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(-imag(bd), real(bd))

			data[i0] = (t0 + t2) * scale_c
			data[i1] = (t1 + t3) * scale_c
			data[i2] = (t0 - t2) * scale_c
			data[i3] = (t1 - t3) * scale_c

			idx1 += stride
			idx2 += stride2
			idx3 += stride3
		}
		return
	}

	base := 0
	idx1 := 0
	idx2 := 0
	idx3 := 0
	k := 0
	when FFT_USE_SIMD_KERNELS {
		for k+1 < quarter {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter

			a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
			bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
			cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
			dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

			idx1n := idx1 + stride
			idx2n := idx2 + stride2
			idx3n := idx3 + stride3
			w1, w2, w3: simd.f64x4
			if use_inv {
				w1 = simd.f64x4{
					real(plan.twiddles_inv[idx1]), imag(plan.twiddles_inv[idx1]),
					real(plan.twiddles_inv[idx1n]), imag(plan.twiddles_inv[idx1n]),
				}
				w2 = simd.f64x4{
					real(plan.twiddles_inv[idx2]), imag(plan.twiddles_inv[idx2]),
					real(plan.twiddles_inv[idx2n]), imag(plan.twiddles_inv[idx2n]),
				}
				tw3a := plan.twiddles_inv[idx3] if idx3 < half_n else -plan.twiddles_inv[idx3-half_n]
				tw3b := plan.twiddles_inv[idx3n] if idx3n < half_n else -plan.twiddles_inv[idx3n-half_n]
				w3 = simd.f64x4{real(tw3a), imag(tw3a), real(tw3b), imag(tw3b)}
			} else {
				t1a := plan.twiddles[idx1]
				t1b := plan.twiddles[idx1n]
				t2a := plan.twiddles[idx2]
				t2b := plan.twiddles[idx2n]
				t3a := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
				t3b := plan.twiddles[idx3n] if idx3n < half_n else -plan.twiddles[idx3n-half_n]
				w1 = simd.f64x4{real(t1a), -imag(t1a), real(t1b), -imag(t1b)}
				w2 = simd.f64x4{real(t2a), -imag(t2a), real(t2b), -imag(t2b)}
				w3 = simd.f64x4{real(t3a), -imag(t3a), real(t3b), -imag(t3b)}
			}

			b := simd_cmul2_f64x4(bv, w1)
			c := simd_cmul2_f64x4(cv, w2)
			d := simd_cmul2_f64x4(dv, w3)

			t0 := simd.add(a, c)
			t1 := simd.sub(a, c)
			t2 := simd.add(b, d)
			bd := simd.sub(b, d)
			t3 := simd_rot_inv_bd(bd)

			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.mul(simd.add(t0, t2), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.mul(simd.add(t1, t3), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.mul(simd.sub(t0, t2), scale_vec))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.mul(simd.sub(t1, t3), scale_vec))

			k += 2
			idx1 += 2 * stride
			idx2 += 2 * stride2
			idx3 += 2 * stride3
		}
	}
	for ; k < quarter; k += 1 {
		i0 := base + k
		i1 := i0 + quarter
		i2 := i1 + quarter
		i3 := i2 + quarter

		a := data[i0]
		w1, w2, w3: complex128
		if use_inv {
			w1 = plan.twiddles_inv[idx1]
			w2 = plan.twiddles_inv[idx2]
			w3 = plan.twiddles_inv[idx3] if idx3 < half_n else -plan.twiddles_inv[idx3-half_n]
		} else {
			t1w := plan.twiddles[idx1]
			t2w := plan.twiddles[idx2]
			t3w := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
			w1 = complex(real(t1w), -imag(t1w))
			w2 = complex(real(t2w), -imag(t2w))
			w3 = complex(real(t3w), -imag(t3w))
		}
		b := data[i1] * w1
		c := data[i2] * w2
		d := data[i3] * w3

		t0 := a + c
		t1 := a - c
		t2 := b + d
		bd := b - d
		t3 := complex(-imag(bd), real(bd))

		data[i0] = (t0 + t2) * scale_c
		data[i1] = (t1 + t3) * scale_c
		data[i2] = (t0 - t2) * scale_c
		data[i3] = (t1 - t3) * scale_c

		idx1 += stride
		idx2 += stride2
		idx3 += stride3
	}
}

cooley_tukey_forward_radix4_stage_chunks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride, k_start, k_end: int,
) {
	if k_start >= k_end {
		return
	}

	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	tw := plan.twiddles
	stage := cooley_tukey_find_radix4_stage(plan, len)
	block_count := plan.n / len

	#no_bounds_check for block := 0; block < block_count; block += 1 {
		base := block * len
		k := k_start

		if stage != nil && stage.tw != nil {
			if (k & 1) != 0 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter
				idx1 := k * stride
				idx2 := k * stride2
				idx3 := k * stride3

				a := data[i0]
				w1 := tw[idx1]
				w2 := tw[idx2]
				w3 := tw[idx3] if idx3 < half_n else -tw[idx3-half_n]
				b := data[i1] * w1
				c := data[i2] * w2
				d := data[i3] * w3

				t0 := a + c
				t1 := a - c
				t2 := b + d
				bd := b - d
				t3 := complex(imag(bd), -real(bd))

				data[i0] = t0 + t2
				data[i1] = t1 + t3
				data[i2] = t0 - t2
				data[i3] = t1 - t3
				k += 1
			}

			p := k / 2
			for ; k+1 < k_end; k += 2 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter

				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				w1 := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1))
				w2 := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2))
				w3 := intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3))

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_fwd_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))
				p += 1
			}
		}

		for ; k < k_end; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter
			idx1 := k * stride
			idx2 := k * stride2
			idx3 := k * stride3

			a := data[i0]
			w1 := tw[idx1]
			w2 := tw[idx2]
			w3 := tw[idx3] if idx3 < half_n else -tw[idx3-half_n]
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(imag(bd), -real(bd))

			data[i0] = t0 + t2
			data[i1] = t1 + t3
			data[i2] = t0 - t2
			data[i3] = t1 - t3
		}
	}
}

cooley_tukey_inverse_radix4_stage_inv_chunks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride, k_start, k_end: int,
) {
	if k_start >= k_end {
		return
	}

	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	tw_inv := plan.twiddles_inv
	stage := cooley_tukey_find_radix4_stage(plan, len)
	block_count := plan.n / len

	#no_bounds_check for block := 0; block < block_count; block += 1 {
		base := block * len
		k := k_start

		if stage != nil && stage.tw != nil {
			if (k & 1) != 0 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter
				idx1 := k * stride
				idx2 := k * stride2
				idx3 := k * stride3

				a := data[i0]
				w1 := tw_inv[idx1]
				w2 := tw_inv[idx2]
				w3 := tw_inv[idx3] if idx3 < half_n else -tw_inv[idx3-half_n]
				b := data[i1] * w1
				c := data[i2] * w2
				d := data[i3] * w3

				t0 := a + c
				t1 := a - c
				t2 := b + d
				bd := b - d
				t3 := complex(-imag(bd), real(bd))

				data[i0] = t0 + t2
				data[i1] = t1 + t3
				data[i2] = t0 - t2
				data[i3] = t1 - t3
				k += 1
			}

			p := k / 2
			for ; k+1 < k_end; k += 2 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter

				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				w1 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1)), SIMD_CONJ_SIGN)
				w2 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2)), SIMD_CONJ_SIGN)
				w3 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3)), SIMD_CONJ_SIGN)

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_inv_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))
				p += 1
			}
		}

		for ; k < k_end; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter
			idx1 := k * stride
			idx2 := k * stride2
			idx3 := k * stride3

			a := data[i0]
			w1 := tw_inv[idx1]
			w2 := tw_inv[idx2]
			w3 := tw_inv[idx3] if idx3 < half_n else -tw_inv[idx3-half_n]
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(-imag(bd), real(bd))

			data[i0] = t0 + t2
			data[i1] = t1 + t3
			data[i2] = t0 - t2
			data[i3] = t1 - t3
		}
	}
}

cooley_tukey_inverse_radix4_stage_conj_chunks :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, stride, k_start, k_end: int,
) {
	if k_start >= k_end {
		return
	}

	quarter := len / 4
	stride2 := stride * 2
	stride3 := stride * 3
	half_n := plan.n / 2
	tw := plan.twiddles
	stage := cooley_tukey_find_radix4_stage(plan, len)
	block_count := plan.n / len

	#no_bounds_check for block := 0; block < block_count; block += 1 {
		base := block * len
		k := k_start

		if stage != nil && stage.tw != nil {
			if (k & 1) != 0 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter
				idx1 := k * stride
				idx2 := k * stride2
				idx3 := k * stride3

				a := data[i0]
				w1 := conj(tw[idx1])
				w2 := conj(tw[idx2])
				w3 := conj(tw[idx3]) if idx3 < half_n else -conj(tw[idx3-half_n])
				b := data[i1] * w1
				c := data[i2] * w2
				d := data[i3] * w3

				t0 := a + c
				t1 := a - c
				t2 := b + d
				bd := b - d
				t3 := complex(-imag(bd), real(bd))

				data[i0] = t0 + t2
				data[i1] = t1 + t3
				data[i2] = t0 - t2
				data[i3] = t1 - t3
				k += 1
			}

			p := k / 2
			for ; k+1 < k_end; k += 2 {
				i0 := base + k
				i1 := i0 + quarter
				i2 := i1 + quarter
				i3 := i2 + quarter

				a := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i0]))
				bv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i1]))
				cv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i2]))
				dv := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i3]))

				w1 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w1)), SIMD_CONJ_SIGN)
				w2 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w2)), SIMD_CONJ_SIGN)
				w3 := simd.mul(intrinsics.unaligned_load(cast(^simd.f64x4)(&stage.tw[p].w3)), SIMD_CONJ_SIGN)

				b := simd_cmul2_f64x4(bv, w1)
				c := simd_cmul2_f64x4(cv, w2)
				d := simd_cmul2_f64x4(dv, w3)

				t0 := simd.add(a, c)
				t1 := simd.sub(a, c)
				t2 := simd.add(b, d)
				bd := simd.sub(b, d)
				t3 := simd_rot_inv_bd(bd)

				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i0]), simd.add(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i1]), simd.add(t1, t3))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i2]), simd.sub(t0, t2))
				intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i3]), simd.sub(t1, t3))
				p += 1
			}
		}

		for ; k < k_end; k += 1 {
			i0 := base + k
			i1 := i0 + quarter
			i2 := i1 + quarter
			i3 := i2 + quarter
			idx1 := k * stride
			idx2 := k * stride2
			idx3 := k * stride3

			a := data[i0]
			w1 := conj(tw[idx1])
			w2 := conj(tw[idx2])
			w3 := conj(tw[idx3]) if idx3 < half_n else -conj(tw[idx3-half_n])
			b := data[i1] * w1
			c := data[i2] * w2
			d := data[i3] * w3

			t0 := a + c
			t1 := a - c
			t2 := b + d
			bd := b - d
			t3 := complex(-imag(bd), real(bd))

			data[i0] = t0 + t2
			data[i1] = t1 + t3
			data[i2] = t0 - t2
			data[i3] = t1 - t3
		}
	}
}

cooley_tukey_parallel_worker_proc :: proc(t: ^thread.Thread) {
	ctx := (^C2C_Parallel_Worker_Context)(t.data)
	plan := ctx.plan
	worker_index := ctx.worker_index

	for {
		if !plan.parallel_enabled {
			return
		}
		sync.barrier_wait(&plan.parallel_start)

		mode := plan.parallel_mode
		if mode == .Stop {
			sync.barrier_wait(&plan.parallel_done)
			return
		}

		len := plan.parallel_len
		worker_count := plan.parallel_worker_count
		switch mode {
		case .Forward_R4_Chunk, .Inverse_R4_Inv_Chunk, .Inverse_R4_Conj_Chunk:
			quarter := len / 4
			chunk := (quarter + worker_count - 1) / worker_count
			k_start := worker_index * chunk
			k_end := min(k_start + chunk, quarter)
			if k_start < k_end {
				switch mode {
				case .Forward_R4_Chunk:
					cooley_tukey_forward_radix4_stage_chunks(plan, plan.parallel_data, len, plan.parallel_stride, k_start, k_end)
				case .Inverse_R4_Inv_Chunk:
					cooley_tukey_inverse_radix4_stage_inv_chunks(plan, plan.parallel_data, len, plan.parallel_stride, k_start, k_end)
				case .Inverse_R4_Conj_Chunk:
					cooley_tukey_inverse_radix4_stage_conj_chunks(plan, plan.parallel_data, len, plan.parallel_stride, k_start, k_end)
				case .Idle, .Forward, .Forward_R4, .Inverse_Inv, .Inverse_R4_Inv, .Inverse_Conj, .Inverse_R4_Conj, .Stop:
				}
			}
		case .Forward, .Forward_R4, .Inverse_Inv, .Inverse_R4_Inv, .Inverse_Conj, .Inverse_R4_Conj:
			block_count := plan.n / len
			chunk := (block_count + worker_count - 1) / worker_count
			block_start := worker_index * chunk
			block_end := min(block_start + chunk, block_count)
			if block_start < block_end {
				switch mode {
				case .Forward:
					cooley_tukey_forward_stage_blocks(plan, plan.parallel_data, len, plan.parallel_half, plan.parallel_stride, block_start, block_end)
				case .Forward_R4:
					cooley_tukey_forward_radix4_stage_blocks(plan, plan.parallel_data, len, plan.parallel_stride, block_start, block_end)
				case .Inverse_Inv:
					cooley_tukey_inverse_stage_inv_blocks(plan, plan.parallel_data, len, plan.parallel_half, plan.parallel_stride, block_start, block_end)
				case .Inverse_R4_Inv:
					cooley_tukey_inverse_radix4_stage_inv_blocks(plan, plan.parallel_data, len, plan.parallel_stride, block_start, block_end)
				case .Inverse_Conj:
					cooley_tukey_inverse_stage_conj_blocks(plan, plan.parallel_data, len, plan.parallel_half, plan.parallel_stride, block_start, block_end)
				case .Inverse_R4_Conj:
					cooley_tukey_inverse_radix4_stage_conj_blocks(plan, plan.parallel_data, len, plan.parallel_stride, block_start, block_end)
				case .Idle, .Forward_R4_Chunk, .Inverse_R4_Inv_Chunk, .Inverse_R4_Conj_Chunk, .Stop:
				}
			}
		case .Idle, .Stop:
		}
		sync.barrier_wait(&plan.parallel_done)
	}
}

cooley_tukey_parallel_init :: proc(plan: ^C2C_Plan) -> Error {
	if plan.num_threads <= 1 || plan.n < COOLEY_TUKEY_PARALLEL_MIN_N {
		plan.num_threads = max(plan.num_threads, 1)
		return .None
	}

	worker_count := min(plan.num_threads, plan.n / COOLEY_TUKEY_PARALLEL_MIN_BLOCKS_PER_WORKER)
	if worker_count <= 1 {
		plan.num_threads = 1
		return .None
	}

	threads, threads_err := make([]^thread.Thread, worker_count, plan.allocator)
	if threads_err != .None {
		return .Allocation_Failed
	}
	ctx, ctx_err := make([]C2C_Parallel_Worker_Context, worker_count, plan.allocator)
	if ctx_err != .None {
		delete(threads, plan.allocator)
		return .Allocation_Failed
	}

	created := 0
	for i in 0..<worker_count {
		ctx[i] = C2C_Parallel_Worker_Context{plan = plan, worker_index = i}
		t := thread.create(cooley_tukey_parallel_worker_proc)
		if t == nil {
			for j in 0..<created {
				thread.destroy(threads[j])
			}
			delete(ctx, plan.allocator)
			delete(threads, plan.allocator)
			return .Allocation_Failed
		}
		t.data = &ctx[i]
		threads[i] = t
		created += 1
	}

	plan.parallel_threads = threads
	plan.parallel_ctx = ctx
	plan.parallel_worker_count = worker_count
	plan.parallel_mode = .Idle
	sync.barrier_init(&plan.parallel_start, worker_count + 1)
	sync.barrier_init(&plan.parallel_done, worker_count + 1)
	plan.parallel_enabled = true

	for t in plan.parallel_threads {
		thread.start(t)
	}
	return .None
}

cooley_tukey_parallel_shutdown :: proc(plan: ^C2C_Plan) {
	if !plan.parallel_enabled {
		if plan.parallel_ctx != nil {
			delete(plan.parallel_ctx, plan.allocator)
		}
		if plan.parallel_threads != nil {
			for t in plan.parallel_threads {
				if t != nil {
					thread.destroy(t)
				}
			}
			delete(plan.parallel_threads, plan.allocator)
		}
		plan.parallel_ctx = nil
		plan.parallel_threads = nil
		plan.parallel_worker_count = 0
		plan.parallel_mode = .Idle
		return
	}

	plan.parallel_mode = .Stop
	sync.barrier_wait(&plan.parallel_start)
	sync.barrier_wait(&plan.parallel_done)

	for t in plan.parallel_threads {
		thread.destroy(t)
	}
	delete(plan.parallel_threads, plan.allocator)
	delete(plan.parallel_ctx, plan.allocator)
	plan.parallel_threads = nil
	plan.parallel_ctx = nil
	plan.parallel_worker_count = 0
	plan.parallel_enabled = false
	plan.parallel_mode = .Idle
}

cooley_tukey_parallel_should_run_stage :: #force_inline proc(plan: ^C2C_Plan, block_count: int) -> bool {
	if !plan.parallel_enabled || plan.parallel_worker_count <= 1 {
		return false
	}
	return block_count >= plan.parallel_worker_count*COOLEY_TUKEY_PARALLEL_MIN_BLOCKS_PER_WORKER
}

cooley_tukey_parallel_should_run_radix4_chunked :: #force_inline proc(plan: ^C2C_Plan, len, block_count: int) -> bool {
	if !plan.parallel_enabled || plan.parallel_worker_count <= 1 {
		return false
	}
	if block_count >= plan.parallel_worker_count*COOLEY_TUKEY_PARALLEL_MIN_BLOCKS_PER_WORKER {
		return false
	}
	return len >= RADIX4_PARALLEL_CHUNK_MIN_LEN
}

cooley_tukey_parallel_run_stage :: proc(
	plan: ^C2C_Plan,
	data: []complex128,
	len, half, stride: int,
	mode: C2C_Parallel_Mode,
) {
	plan.parallel_data = data
	plan.parallel_len = len
	plan.parallel_half = half
	plan.parallel_stride = stride
	plan.parallel_mode = mode
	sync.barrier_wait(&plan.parallel_start)
	sync.barrier_wait(&plan.parallel_done)
	plan.parallel_mode = .Idle
}

// Iterative radix-2 Cooley-Tukey with bit-reversal permutation.
cooley_tukey_bit_reverse_permute_in_place :: proc(plan: ^C2C_Plan, data: []complex128) {
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

	// Low-RAM path: compute bit-reversal permutation on the fly.
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

cooley_tukey_digit_reverse4_permute_in_place :: proc(plan: ^C2C_Plan, data: []complex128) {
	n := plan.n
	if plan.digitrev4 != nil {
		if n <= 256 {
			#no_bounds_check for i in 0..<n {
				j := int(plan.digitrev4[i])
				if j > i {
					data[i], data[j] = data[j], data[i]
				}
			}
			return
		}
		#no_bounds_check for i in 0..<n {
			j := int(plan.digitrev4[i])
			if j > i {
				data[i], data[j] = data[j], data[i]
			}
		}
		return
	}

	digit_count := plan.log2_n / 2
	#no_bounds_check for i in 0..<n {
		j := int(reverse_base4_u32(u32(i), digit_count))
		if j > i {
			data[i], data[j] = data[j], data[i]
		}
	}
}

cooley_tukey_forward_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
		return .None
	}

	cooley_tukey_bit_reverse_permute_in_place(plan, data)

	// len=2 stage (twiddle is 1)
	i := 0
	when FFT_USE_SIMD_KERNELS {
		for ; i + 1 < n; i += 2 {
			v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i]))
			u := simd.shuffle(v, v, 0, 1, 0, 1)
			w := simd.shuffle(v, v, 2, 3, 2, 3)
			sum := simd.add(u, w)
			diff := simd.sub(u, w)
			res := simd.shuffle(sum, diff, 0, 1, 4, 5)
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i]), res)
		}
	}
	for ; i < n; i += 2 {
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
		block_count := n / len
		if cooley_tukey_parallel_should_run_stage(plan, block_count) {
			cooley_tukey_parallel_run_stage(plan, data, len, half, stride, .Forward)
		} else {
			cooley_tukey_forward_stage_blocks(plan, data, len, half, stride, 0, block_count)
		}
		len <<= 1
	}

	return .None
}

cooley_tukey_inverse_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
		return .None
	}

	cooley_tukey_bit_reverse_permute_in_place(plan, data)

	i := 0
	when FFT_USE_SIMD_KERNELS {
		for ; i + 1 < n; i += 2 {
			v := intrinsics.unaligned_load(cast(^simd.f64x4)(&data[i]))
			u := simd.shuffle(v, v, 0, 1, 0, 1)
			w := simd.shuffle(v, v, 2, 3, 2, 3)
			sum := simd.add(u, w)
			diff := simd.sub(u, w)
			res := simd.shuffle(sum, diff, 0, 1, 4, 5)
			intrinsics.unaligned_store(cast(^simd.f64x4)(&data[i]), res)
		}
	}
	for ; i < n; i += 2 {
		a := data[i]
		b := data[i+1]
		data[i] = a + b
		data[i+1] = a - b
	}

	if n > 2 {
		inv_n := 1.0 / f64(n)
		len := 4
		if !plan.parallel_enabled {
			if plan.twiddles_inv != nil {
				for len < n {
					half := len / 2
					stride := n / len
					cooley_tukey_inverse_stage_inv_blocks(plan, data, len, half, stride, 0, n/len)
					len <<= 1
				}
				cooley_tukey_inverse_stage_scaled(plan, data, n, n/2, 1, inv_n, true)
			} else {
				for len < n {
					half := len / 2
					stride := n / len
					cooley_tukey_inverse_stage_conj_blocks(plan, data, len, half, stride, 0, n/len)
					len <<= 1
				}
				cooley_tukey_inverse_stage_scaled(plan, data, n, n/2, 1, inv_n, false)
			}
			return .None
		}
		if plan.twiddles_inv != nil {
			for len < n {
				half := len / 2
				stride := n / len
				block_count := n / len
				if cooley_tukey_parallel_should_run_stage(plan, block_count) {
					cooley_tukey_parallel_run_stage(plan, data, len, half, stride, .Inverse_Inv)
				} else {
					cooley_tukey_inverse_stage_inv_blocks(plan, data, len, half, stride, 0, block_count)
				}
				len <<= 1
			}
			cooley_tukey_inverse_stage_scaled(plan, data, n, n/2, 1, inv_n, true)
		} else {
			for len < n {
				half := len / 2
				stride := n / len
				block_count := n / len
				if cooley_tukey_parallel_should_run_stage(plan, block_count) {
					cooley_tukey_parallel_run_stage(plan, data, len, half, stride, .Inverse_Conj)
				} else {
					cooley_tukey_inverse_stage_conj_blocks(plan, data, len, half, stride, 0, block_count)
				}
				len <<= 1
			}
			cooley_tukey_inverse_stage_scaled(plan, data, n, n/2, 1, inv_n, false)
		}
		return .None
	}

	inv_n := 1.0 / f64(n)
	scale_complex_array_in_place(data, inv_n)
	return .None
}

cooley_tukey_forward_radix4_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
		return .None
	}
	if (plan.log2_n & 1) != 0 {
		return cooley_tukey_forward_in_place(plan, data)
	}

	cooley_tukey_digit_reverse4_permute_in_place(plan, data)

	if !plan.parallel_enabled && n <= 256 {
		for len := 4; len <= n; len *= 4 {
			stride := n / len
			cooley_tukey_forward_radix4_stage_blocks(plan, data, len, stride, 0, n/len)
		}
		return .None
	}

	len := 4
	for len <= n {
		stride := n / len
		block_count := n / len
		if cooley_tukey_parallel_should_run_stage(plan, block_count) {
			cooley_tukey_parallel_run_stage(plan, data, len, len/2, stride, .Forward_R4)
		} else if cooley_tukey_parallel_should_run_radix4_chunked(plan, len, block_count) {
			cooley_tukey_parallel_run_stage(plan, data, len, len/2, stride, .Forward_R4_Chunk)
		} else {
			cooley_tukey_forward_radix4_stage_blocks(plan, data, len, stride, 0, block_count)
		}
		len *= 4
	}

	return .None
}

cooley_tukey_inverse_radix4_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
		return .None
	}
	if (plan.log2_n & 1) != 0 {
		return cooley_tukey_inverse_in_place(plan, data)
	}

	cooley_tukey_digit_reverse4_permute_in_place(plan, data)

	len := 4
	tw_inv := plan.twiddles_inv
	inv_n := 1.0 / f64(n)
	if !plan.parallel_enabled && n <= 256 {
		for len < n {
			stride := n / len
			if tw_inv != nil {
				cooley_tukey_inverse_radix4_stage_inv_blocks(plan, data, len, stride, 0, n/len)
			} else {
				cooley_tukey_inverse_radix4_stage_conj_blocks(plan, data, len, stride, 0, n/len)
			}
			len *= 4
		}
		cooley_tukey_inverse_radix4_stage_scaled(plan, data, n, 1, inv_n, tw_inv != nil)
		return .None
	}
	if !plan.parallel_enabled {
		for len < n {
			stride := n / len
			if tw_inv != nil {
				cooley_tukey_inverse_radix4_stage_inv_blocks(plan, data, len, stride, 0, n/len)
			} else {
				cooley_tukey_inverse_radix4_stage_conj_blocks(plan, data, len, stride, 0, n/len)
			}
			len *= 4
		}
		cooley_tukey_inverse_radix4_stage_scaled(plan, data, n, 1, inv_n, tw_inv != nil)
		return .None
	}

	for len <= n {
		stride := n / len
		block_count := n / len
		if tw_inv != nil {
			if cooley_tukey_parallel_should_run_stage(plan, block_count) {
				cooley_tukey_parallel_run_stage(plan, data, len, len/2, stride, .Inverse_R4_Inv)
			} else if cooley_tukey_parallel_should_run_radix4_chunked(plan, len, block_count) {
				cooley_tukey_parallel_run_stage(plan, data, len, len/2, stride, .Inverse_R4_Inv_Chunk)
			} else {
				cooley_tukey_inverse_radix4_stage_inv_blocks(plan, data, len, stride, 0, block_count)
			}
		} else {
			if cooley_tukey_parallel_should_run_stage(plan, block_count) {
				cooley_tukey_parallel_run_stage(plan, data, len, len/2, stride, .Inverse_R4_Conj)
			} else if cooley_tukey_parallel_should_run_radix4_chunked(plan, len, block_count) {
				cooley_tukey_parallel_run_stage(plan, data, len, len/2, stride, .Inverse_R4_Conj_Chunk)
			} else {
				cooley_tukey_inverse_radix4_stage_conj_blocks(plan, data, len, stride, 0, block_count)
			}
		}
		len *= 4
	}

	scale_complex_array_in_place(data, inv_n)
	return .None
}

cooley_tukey_forward_radix8_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	if plan.n == 8 {
		data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7] =
			cooley_tukey_dft8_from_bitrev_forward(
				data[0], data[4], data[2], data[6],
				data[1], data[5], data[3], data[7],
			)
		return .None
	}
	if plan.parallel_enabled || plan.log2_n < 3 || ((plan.log2_n - 3) & 1) != 0 {
		return cooley_tukey_forward_radix4_in_place(plan, data)
	}
	cooley_tukey_digit_reverse8_permute_in_place(plan, data)
	cooley_tukey_forward_radix8_base_blocks(data, 0, plan.n/8)
	for len := 32; len <= plan.n; len *= 4 {
		stride := plan.n / len
		cooley_tukey_forward_radix4_stage_blocks(plan, data, len, stride, 0, plan.n/len)
	}
	return cooley_tukey_forward_radix4_in_place(plan, data)
}

cooley_tukey_inverse_radix8_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	if plan.n == 8 {
		data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7] =
			cooley_tukey_dft8_from_bitrev_inverse(
				data[0], data[4], data[2], data[6],
				data[1], data[5], data[3], data[7],
			)
		scale_complex_array_in_place(data, 1.0 / 8.0)
		return .None
	}
	if plan.parallel_enabled || plan.log2_n < 3 || ((plan.log2_n - 3) & 1) != 0 {
		return cooley_tukey_inverse_radix4_in_place(plan, data)
	}
	cooley_tukey_digit_reverse8_permute_in_place(plan, data)
	cooley_tukey_inverse_radix8_base_blocks(data, 0, plan.n/8)
	tw_inv := plan.twiddles_inv
	for len := 32; len <= plan.n; len *= 4 {
		stride := plan.n / len
		if tw_inv != nil {
			cooley_tukey_inverse_radix4_stage_inv_blocks(plan, data, len, stride, 0, plan.n/len)
		} else {
			cooley_tukey_inverse_radix4_stage_conj_blocks(plan, data, len, stride, 0, plan.n/len)
		}
	}
	scale_complex_array_in_place(data, 1.0 / f64(plan.n))
	return .None
}

cooley_tukey_forward_radix16_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	return cooley_tukey_forward_radix4_in_place(plan, data)
}

cooley_tukey_inverse_radix16_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	return cooley_tukey_inverse_radix4_in_place(plan, data)
}
