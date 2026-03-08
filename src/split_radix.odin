package fft

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

		e0, e1, e2, e3 := split_radix_dft4_forward(x0, x2, x4, x6)
		o0, o1, o2, o3 := split_radix_dft4_forward(x1, x3, x5, x7)

		c := 0.7071067811865475244
		w1 := complex(c, -c)
		w2 := complex(0.0, -1.0)
		w3 := complex(-c, -c)

		t1 := w1 * o1
		t2 := w2 * o2
		t3 := w3 * o3

		dst[0] = e0 + o0
		dst[4] = e0 - o0
		dst[1] = e1 + t1
		dst[5] = e1 - t1
		dst[2] = e2 + t2
		dst[6] = e2 - t2
		dst[3] = e3 + t3
		dst[7] = e3 - t3
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

split_radix_forward_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	n := plan.n
	if n <= 1 {
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
	if len(plan.scratch) < n {
		return .Size_Mismatch
	}

	scratch := plan.scratch[:n]

	// IFFT(x) = conj(FFT(conj(x))) / N.
	// Write conj(input) into scratch, run forward split-radix from scratch->data,
	// then apply final conjugate+scale in-place.
	#no_bounds_check for i in 0..<n {
		scratch[i] = conj(data[i])
	}

	split_radix_fft_from_strided(plan, data, scratch, n, 1)

	inv_n := 1.0 / f64(n)
	#no_bounds_check for i in 0..<n {
		data[i] = conj(data[i]) * complex(inv_n, 0.0)
	}

	return .None
}
