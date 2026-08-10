package fft

SPLIT_W8_FWD_1 :: complex(0.7071067811865475244, -0.7071067811865475244)
SPLIT_W8_FWD_2 :: complex(0.0, -1.0)
SPLIT_W8_FWD_3 :: complex(-0.7071067811865475244, -0.7071067811865475244)
SPLIT_W8_INV_1 :: complex(0.7071067811865475244, 0.7071067811865475244)
SPLIT_W8_INV_2 :: complex(0.0, 1.0)
SPLIT_W8_INV_3 :: complex(-0.7071067811865475244, 0.7071067811865475244)

split_radix_mul_pos_i :: #force_inline proc(x: complex128) -> complex128 {
	return complex(-imag(x), real(x))
}

split_radix_mul_w8_fwd_1 :: #force_inline proc(x: complex128) -> complex128 {
	s := 0.7071067811865475244
	return complex((real(x) + imag(x)) * s, (imag(x) - real(x)) * s)
}

split_radix_mul_w8_fwd_3 :: #force_inline proc(x: complex128) -> complex128 {
	s := 0.7071067811865475244
	return complex((imag(x) - real(x)) * s, -(real(x) + imag(x)) * s)
}

split_radix_mul_w8_inv_1 :: #force_inline proc(x: complex128) -> complex128 {
	s := 0.7071067811865475244
	return complex((real(x) - imag(x)) * s, (real(x) + imag(x)) * s)
}

split_radix_mul_w8_inv_3 :: #force_inline proc(x: complex128) -> complex128 {
	s := 0.7071067811865475244
	return complex(-(real(x) + imag(x)) * s, (real(x) - imag(x)) * s)
}

split_radix_dft4_forward :: #force_inline proc(x0, x1, x2, x3: complex128) -> (y0, y1, y2, y3: complex128) {
	a0 := x0 + x2
	a1 := x0 - x2
	b0 := x1 + x3
	b1 := x1 - x3
	// -i * b1
	b1m := complex(imag(b1), -real(b1))

	y0 = a0 + b0
	y2 = a0 - b0
	y1 = a1 + b1m
	y3 = a1 - b1m
	return
}

split_radix_dft4_inverse :: #force_inline proc(x0, x1, x2, x3: complex128) -> (y0, y1, y2, y3: complex128) {
	a0 := x0 + x2
	a1 := x0 - x2
	b0 := x1 + x3
	b1 := x1 - x3
	// +i * b1
	b1p := complex(-imag(b1), real(b1))

	y0 = a0 + b0
	y2 = a0 - b0
	y1 = a1 + b1p
	y3 = a1 - b1p
	return
}

split_radix_dft8_forward :: #force_inline proc(
	x0, x1, x2, x3, x4, x5, x6, x7: complex128,
) -> (
	y0, y1, y2, y3, y4, y5, y6, y7: complex128,
) {
	e0, e1, e2, e3 := split_radix_dft4_forward(x0, x2, x4, x6)
	o0, o1, o2, o3 := split_radix_dft4_forward(x1, x3, x5, x7)

	c := 0.7071067811865475244
	w1 := complex(c, -c)
	w2 := complex(0.0, -1.0)
	w3 := complex(-c, -c)

	t1 := w1 * o1
	t2 := w2 * o2
	t3 := w3 * o3

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

split_radix_dft8_inverse :: #force_inline proc(
	x0, x1, x2, x3, x4, x5, x6, x7: complex128,
) -> (
	y0, y1, y2, y3, y4, y5, y6, y7: complex128,
) {
	e0, e1, e2, e3 := split_radix_dft4_inverse(x0, x2, x4, x6)
	o0, o1, o2, o3 := split_radix_dft4_inverse(x1, x3, x5, x7)

	t1 := split_radix_mul_w8_inv_1(o1)
	t2 := complex(-imag(o2), real(o2))
	t3 := split_radix_mul_w8_inv_3(o3)

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

split_radix_fft_from_strided :: proc(
	plan: ^C2C_Plan,
	dst: []complex128,
	src: []complex128,
	n, src_stride: int,
) {
	if n == 1 {
		dst[0] = src[0]
		return
	}
	if n == 2 {
		a := src[0]
		b := src[src_stride]
		dst[0] = a + b
		dst[1] = a - b
		return
	}
	if n == 4 {
		x0 := src[0]
		x1 := src[src_stride]
		x2 := src[2*src_stride]
		x3 := src[3*src_stride]
		dst[0], dst[1], dst[2], dst[3] = split_radix_dft4_forward(x0, x1, x2, x3)
		return
	}
	if n == 8 {
		x0 := src[0]
		x1 := src[src_stride]
		x2 := src[2*src_stride]
		x3 := src[3*src_stride]
		x4 := src[4*src_stride]
		x5 := src[5*src_stride]
		x6 := src[6*src_stride]
		x7 := src[7*src_stride]

		dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7] =
			split_radix_dft8_forward(x0, x1, x2, x3, x4, x5, x6, x7)
		return
	}

	n2 := n / 2
	n4 := n / 4

	split_radix_fft_from_strided(plan, dst[0:n2], src, n2, src_stride*2)
	split_radix_fft_from_strided(plan, dst[n2:][:n4], src[src_stride:], n4, src_stride*4)
	split_radix_fft_from_strided(plan, dst[n2+n4:][:n4], src[3*src_stride:], n4, src_stride*4)

	step := plan.n / n
	half_n := plan.n / 2
	j := complex(0.0, 1.0)
	idx1 := 0
	idx3 := 0
	#no_bounds_check for k in 0..<n4 {
		e0 := dst[k]
		e1 := dst[k+n4]
		o1 := dst[n2+k]
		o3 := dst[n2+n4+k]

		w1 := plan.twiddles[idx1]
		w3 := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]

		t1 := w1 * o1
		t2 := w3 * o3
		tdiff := t1 - t2
		tsum := t1 + t2

		dst[k] = e0 + tsum
		dst[k+n2] = e0 - tsum
		dst[k+n4] = e1 - j*tdiff
		dst[k+3*n4] = e1 + j*tdiff

		idx1 += step
		idx3 += 3 * step
	}
}

split_radix_ifft_from_strided :: proc(
	plan: ^C2C_Plan,
	dst: []complex128,
	src: []complex128,
	n, src_stride: int,
) {
	if n == 1 {
		dst[0] = src[0]
		return
	}
	if n == 2 {
		a := src[0]
		b := src[src_stride]
		dst[0] = a + b
		dst[1] = a - b
		return
	}
	if n == 4 {
		x0 := src[0]
		x1 := src[src_stride]
		x2 := src[2*src_stride]
		x3 := src[3*src_stride]
		dst[0], dst[1], dst[2], dst[3] = split_radix_dft4_inverse(x0, x1, x2, x3)
		return
	}
	if n == 8 {
		x0 := src[0]
		x1 := src[src_stride]
		x2 := src[2*src_stride]
		x3 := src[3*src_stride]
		x4 := src[4*src_stride]
		x5 := src[5*src_stride]
		x6 := src[6*src_stride]
		x7 := src[7*src_stride]

		dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7] =
			split_radix_dft8_inverse(x0, x1, x2, x3, x4, x5, x6, x7)
		return
	}

	n2 := n / 2
	n4 := n / 4

	split_radix_ifft_from_strided(plan, dst[0:n2], src, n2, src_stride*2)
	split_radix_ifft_from_strided(plan, dst[n2:][:n4], src[src_stride:], n4, src_stride*4)
	split_radix_ifft_from_strided(plan, dst[n2+n4:][:n4], src[3*src_stride:], n4, src_stride*4)

	step := plan.n / n
	half_n := plan.n / 2
	idx1 := 0
	idx3 := 0
	tw_inv := plan.twiddles_inv
	if tw_inv == nil {
		#no_bounds_check for k in 0..<n4 {
			e0 := dst[k]
			e1 := dst[k+n4]
			o1 := dst[n2+k]
			o3 := dst[n2+n4+k]

			w1 := conj(plan.twiddles[idx1])
			t3 := plan.twiddles[idx3] if idx3 < half_n else -plan.twiddles[idx3-half_n]
			w3 := conj(t3)

			t1 := w1 * o1
			t2 := w3 * o3
			tdiff := t1 - t2
			tsum := t1 + t2
			trot := split_radix_mul_pos_i(tdiff)

			dst[k] = e0 + tsum
			dst[k+n2] = e0 - tsum
			dst[k+n4] = e1 + trot
			dst[k+3*n4] = e1 - trot

			idx1 += step
			idx3 += 3 * step
		}
		return
	}
	#no_bounds_check for k in 0..<n4 {
		e0 := dst[k]
		e1 := dst[k+n4]
		o1 := dst[n2+k]
		o3 := dst[n2+n4+k]

		w1 := tw_inv[idx1]
		w3 := tw_inv[idx3] if idx3 < half_n else -tw_inv[idx3-half_n]

		t1 := w1 * o1
		t2 := w3 * o3
		tdiff := t1 - t2
		tsum := t1 + t2
		trot := split_radix_mul_pos_i(tdiff)

		dst[k] = e0 + tsum
		dst[k+n2] = e0 - tsum
		dst[k+n4] = e1 + trot
		dst[k+3*n4] = e1 - trot

		idx1 += step
		idx3 += 3 * step
	}
}

split_radix_forward_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
		return .None
	}
	if n == 2 {
		a := data[0]
		b := data[1]
		data[0] = a + b
		data[1] = a - b
		return .None
	}
	if n == 4 {
		data[0], data[1], data[2], data[3] =
			split_radix_dft4_forward(data[0], data[1], data[2], data[3])
		return .None
	}
	if n == 8 {
		data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7] =
			split_radix_dft8_forward(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
		return .None
	}
	if len(plan.scratch) < n {
		return .Size_Mismatch
	}

	scratch := plan.scratch[:n]
	split_radix_fft_from_strided(plan, scratch, data, n, 1)
	copy(data, scratch)
	return .None
}

split_radix_inverse_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
		return .None
	}
	if n == 2 {
		a := data[0]
		b := data[1]
		data[0] = (a + b) * 0.5
		data[1] = (a - b) * 0.5
		return .None
	}
	if n == 4 {
		data[0], data[1], data[2], data[3] =
			split_radix_dft4_inverse(data[0], data[1], data[2], data[3])
		scale_complex_array_in_place(data, 1.0 / 4.0)
		return .None
	}
	if n == 8 {
		data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7] =
			split_radix_dft8_inverse(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
		scale_complex_array_in_place(data, 1.0 / 8.0)
		return .None
	}
	if len(plan.scratch) < n {
		return .Size_Mismatch
	}

	scratch := plan.scratch[:n]
	split_radix_ifft_from_strided(plan, scratch, data, n, 1)

	scale_copy_complex_array(data, scratch, 1.0 / f64(n))

	return .None
}
