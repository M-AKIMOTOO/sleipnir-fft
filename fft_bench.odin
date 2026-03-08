package main

import "core:fmt"
import "core:math"
import "core:time"
import fft "sleipnirfft:src"

bench_sizes :: []int{1024, 4096, 16384, 65536, 262144}

pick_iterations :: proc(n: int) -> int {
	target_points := 1 << 22
	it := target_points / n
	if it < 4 {
		it = 4
	}
	return it
}

fill_complex_signal :: proc(dst: []complex128) {
	for i in 0..<len(dst) {
		x := f64(i)
		dst[i] = complex(
			0.4*math.sin(2.0*math.PI*x/17.0)+0.2*math.cos(2.0*math.PI*x/31.0),
			0.5*math.sin(2.0*math.PI*x/23.0)-0.3*math.cos(2.0*math.PI*x/29.0),
		)
	}
}

fill_real_signal :: proc(dst: []f64) {
	for i in 0..<len(dst) {
		x := f64(i)
		dst[i] = math.sin(2.0*math.PI*x/19.0) + 0.25*math.cos(2.0*math.PI*x/41.0)
	}
}

bench_c2c :: proc(n: int) {
	iterations := pick_iterations(n)

	plan: fft.C2C_Plan
	if err := fft.c2c_plan_init(&plan, n); err != .None {
		fmt.printf("c2c n=%d: plan init failed: %v\n", n, err)
		return
	}
	defer fft.c2c_plan_destroy(&plan)

	buf, buf_err := make([]complex128, n)
	orig, orig_err := make([]complex128, n)
	if buf_err != .None || orig_err != .None {
		fmt.printf("c2c n=%d: allocation failed: %v / %v\n", n, buf_err, orig_err)
		return
	}
	defer delete(buf)
	defer delete(orig)

	fill_complex_signal(orig)
	copy(buf, orig)

	_ = fft.c2c_forward_in_place(&plan, buf)
	_ = fft.c2c_inverse_in_place(&plan, buf)

	sink := 0.0
	start := time.now()
	for _ in 0..<iterations {
		copy(buf, orig)
		_ = fft.c2c_forward_in_place(&plan, buf)
		_ = fft.c2c_inverse_in_place(&plan, buf)
		sink += real(buf[0])
	}
	elapsed := time.since(start)

	seconds := f64(time.duration_nanoseconds(elapsed)) / 1e9
	transforms := f64(iterations * 2)
	points_per_sec := f64(n) * transforms / seconds
	ns_per_transform := f64(time.duration_nanoseconds(elapsed)) / transforms

	fmt.printf(
		"c2c n=%7d iter=%6d ns/fft=%10.1f points/s=%12.0f sink=%f\n",
		n, iterations, ns_per_transform, points_per_sec, sink,
	)
}

bench_r2c :: proc(n: int) {
	iterations := pick_iterations(n)

	plan: fft.R2C_Plan
	if err := fft.r2c_plan_init(&plan, n); err != .None {
		fmt.printf("r2c n=%d: plan init failed: %v\n", n, err)
		return
	}
	defer fft.r2c_plan_destroy(&plan)

	input, in_err := make([]f64, n)
	freq, fq_err := make([]complex128, n/2+1)
	out, out_err := make([]f64, n)
	if in_err != .None || fq_err != .None || out_err != .None {
		fmt.printf("r2c n=%d: allocation failed: %v / %v / %v\n", n, in_err, fq_err, out_err)
		return
	}
	defer delete(input)
	defer delete(freq)
	defer delete(out)

	fill_real_signal(input)

	_ = fft.r2c_forward(&plan, input, freq)
	_ = fft.c2r_inverse(&plan, freq, out)

	sink := 0.0
	start := time.now()
	for _ in 0..<iterations {
		_ = fft.r2c_forward(&plan, input, freq)
		_ = fft.c2r_inverse(&plan, freq, out)
		sink += out[0]
	}
	elapsed := time.since(start)

	seconds := f64(time.duration_nanoseconds(elapsed)) / 1e9
	transforms := f64(iterations * 2)
	points_per_sec := f64(n) * transforms / seconds
	ns_per_transform := f64(time.duration_nanoseconds(elapsed)) / transforms

	fmt.printf(
		"r2c n=%7d iter=%6d ns/fft=%10.1f points/s=%12.0f sink=%f\n",
		n, iterations, ns_per_transform, points_per_sec, sink,
	)
}

main :: proc() {
	fmt.println("FFT benchmark (forward+inverse, plan reused)")
	for n in bench_sizes {
		bench_c2c(n)
	}
	fmt.println("")
	for n in bench_sizes {
		bench_r2c(n)
	}
}
