package fft

import "base:runtime"
import "core:math"
import "core:os"
import "core:sync"
import "core:thread"
import "core:time"

VERSION_MAJOR :: 0
VERSION_MINOR :: 1
VERSION_PATCH :: 0
VERSION_STRING :: "0.1.0"

Error :: enum {
	None,
	Plan_Not_Initialized,
	Invalid_Length,
	Length_Not_Power_Of_Two,
	Size_Mismatch,
	Allocation_Failed,
}

Backend :: enum {
	Cooley_Tukey,
	Split_Radix,
	Auto,
}

C2C_Plan_Options :: struct {
	backend:               Backend,
	store_inverse_twiddles: bool,
	store_bitrev_table:    bool,
	threads:               int,  // <=0 means auto
	cooley_radix:          int,  // 0=auto, 2=radix-2, 4=radix-4
}

C2C_Parallel_Mode :: enum {
	Idle,
	Forward,
	Forward_R4,
	Inverse_Inv,
	Inverse_R4_Inv,
	Inverse_Conj,
	Inverse_R4_Conj,
	Stop,
}

C2C_Parallel_Worker_Context :: struct {
	plan:         ^C2C_Plan,
	worker_index: int,
}

Radix4_Twiddle_Pair_Pack :: struct {
	w0_re: f64,
	w0_im: f64,
	w1_re: f64,
	w1_im: f64,
}

Radix2_Twiddle_Pair_Pack :: struct {
	w0_re: f64,
	w0_im: f64,
	w1_re: f64,
	w1_im: f64,
}

Radix2_SIMD_Stage :: struct {
	len: int,
	tw: []Radix2_Twiddle_Pair_Pack,
	tw_inv: []Radix2_Twiddle_Pair_Pack,
}

Radix4_SIMD_Stage :: struct {
	len: int,
	tw1: []Radix4_Twiddle_Pair_Pack,
	tw2: []Radix4_Twiddle_Pair_Pack,
	tw3: []Radix4_Twiddle_Pair_Pack,
}

C2C_Plan :: struct {
	n:           int,
	log2_n:      int,
	backend:     Backend,
	uses_bluestein: bool,
	store_inverse_twiddles: bool,
	store_bitrev_table: bool,
	num_threads: int,
	cooley_radix: int,
	bitrev:      []u32,
	digitrev4:   []u32,
	twiddles:    []complex128, // W_N^k for k in [0, N/2)
	twiddles_inv: []complex128, // conj(W_N^k) for k in [0, N/2)
	scratch:     []complex128, // Used by backends that need temporary space
	parallel_enabled: bool,
	parallel_worker_count: int,
	parallel_threads: []^thread.Thread,
	parallel_ctx: []C2C_Parallel_Worker_Context,
	parallel_start: sync.Barrier,
	parallel_done: sync.Barrier,
	parallel_mode: C2C_Parallel_Mode,
	parallel_data: []complex128,
	parallel_len: int,
	parallel_half: int,
	parallel_stride: int,
	radix2_simd_stages: []Radix2_SIMD_Stage,
	radix4_simd_stages: []Radix4_SIMD_Stage,
	bluestein_m: []complex128,        // work buffer length M
	bluestein_chirp: []complex128,    // exp(-i*pi*n^2/N), length N
	bluestein_b_fft: []complex128,    // FFT of bluestein kernel, length M
	bluestein_conv_plan: ^C2C_Plan,   // power-of-two convolution plan
	allocator:   runtime.Allocator,
	initialized: bool,
}

R2C_Plan :: struct {
	n:           int,
	half_n:      int,
	uses_full_c2c: bool,
	c2c:         C2C_Plan,
	twiddles:    []complex128,
	scratch:     []complex128,
	allocator:   runtime.Allocator,
	initialized: bool,
}

C2C_Plan_2D :: struct {
	rows:        int,
	cols:        int,
	backend:     Backend,
	row_plan:    C2C_Plan,
	col_plan:    C2C_Plan,
	scratch:     []complex128,
	allocator:   runtime.Allocator,
	initialized: bool,
}

is_power_of_two :: proc(n: int) -> bool {
	return n > 0 && (n & (n - 1)) == 0
}

r2c_output_len :: proc(n: int) -> int {
	if n < 2 {
		return 0
	}
	return n/2 + 1
}

reverse_bits_u32 :: #force_inline proc(x: u32, bit_count: int) -> u32 {
	r: u32 = 0
	#no_bounds_check for i in 0..<bit_count {
		r = (r << 1) | ((x >> u32(i)) & 1)
	}
	return r
}

reverse_base4_u32 :: #force_inline proc(x: u32, digit_count: int) -> u32 {
	r: u32 = 0
	v := x
	#no_bounds_check for _ in 0..<digit_count {
		r = (r << 2) | (v & 0b11)
		v >>= 2
	}
	return r
}

log2_exact :: #force_inline proc(n: int) -> int {
	v := n
	log2_n := 0
	for v > 1 {
		log2_n += 1
		v >>= 1
	}
	return log2_n
}

resolve_thread_count :: proc(requested, n: int) -> int {
	if requested > 0 {
		return requested
	}
	if n < (1 << 16) {
		return 1
	}
	core_count := os.get_processor_core_count()
	if core_count <= 1 {
		return 1
	}
	return min(core_count, 4)
}

resolve_cooley_radix :: proc(n, log2_n, requested, num_threads: int) -> int {
	if requested == 2 {
		return 2
	}
	if requested == 4 {
		if (log2_n & 1) == 0 {
			return 4
		}
		return 2
	}
	// Auto heuristic:
	_ = num_threads
	// radix-4 only when length is a power of 4 and size is large enough.
	if n >= 1024 && (log2_n & 1) == 0 {
		return 4
	}
	return 2
}

Auto_C2C_Candidate :: struct {
	backend: Backend,
	radix:   int,
	threads: int,
}

autotune_reps_for_n :: #force_inline proc(n: int) -> int {
	if n <= 1 << 12 {
		return 8
	}
	if n <= 1 << 14 {
		return 7
	}
	if n <= 1 << 16 {
		return 6
	}
	if n <= 1 << 17 {
		return 5
	}
	return 4
}

autotune_samples_for_n :: #force_inline proc(n: int) -> int {
	if n <= 1 << 15 {
		return 4
	}
	if n <= 1 << 17 {
		return 3
	}
	return 2
}

autotune_switch_margin_for_n :: #force_inline proc(n: int) -> f64 {
	// Require a minimum relative win before switching to avoid noise-driven flips.
	if n <= 1 << 14 {
		return 0.02
	}
	if n <= 1 << 16 {
		return 0.03
	}
	return 0.05
}

autotune_stable_fast_ns :: proc(values: []i64) -> i64 {
	count := len(values)
	if count <= 0 {
		return 0
	}
	sorted: [4]i64
	#no_bounds_check for i in 0..<count {
		sorted[i] = values[i]
	}
	for i in 1..<count {
		v := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > v {
			sorted[j+1] = sorted[j]
			j -= 1
		}
		sorted[j+1] = v
	}
	if count <= 2 {
		return sorted[0]
	}
	if count == 3 {
		return sorted[1]
	}
	// 4 samples: choose 2nd fastest (stable but still speed-oriented).
	return sorted[1]
}

autotune_candidate_score :: #force_inline proc(n: int, cand: Auto_C2C_Candidate) -> int {
	score := 0
	thread_delta := max(cand.threads-1, 0)
	if n < 1 << 16 {
		// Small sizes: thread overhead often dominates.
		score += thread_delta * 4
	} else {
		// Larger sizes: allow parallel variants to win tie-breaks.
		score -= min(thread_delta, 3)
	}
	if n <= 1 << 13 {
		// For smaller sizes, split-radix often wins on this implementation.
		if cand.backend == .Split_Radix {
			score -= 2
		}
	}
	if cand.backend == .Cooley_Tukey {
		if cand.radix == 4 {
			score -= 1
		}
	} else {
		score += 1
	}
	return score
}

autotune_prefer_candidate :: #force_inline proc(n: int, cand, best: Auto_C2C_Candidate) -> bool {
	return autotune_candidate_score(n, cand) < autotune_candidate_score(n, best)
}

auto_candidate_equal :: #force_inline proc(a, b: Auto_C2C_Candidate) -> bool {
	return a.backend == b.backend && a.radix == b.radix && a.threads == b.threads
}

auto_candidate_push_unique :: #force_inline proc(candidates: ^[8]Auto_C2C_Candidate, candidate_count: ^int, cand: Auto_C2C_Candidate) {
	count := candidate_count^
	#no_bounds_check for i in 0..<count {
		if auto_candidate_equal(candidates[i], cand) {
			return
		}
	}
	if count >= len(candidates^) {
		return
	}
	candidates[count] = cand
	candidate_count^ = count + 1
}

autotune_fill_signal :: proc(data: []complex128) {
	#no_bounds_check for i in 0..<len(data) {
		re := f64(i & 31) * 0.03125
		im := f64((i * 3) & 31) * 0.03125
		data[i] = complex(re, im)
	}
}

autotune_measure_candidate :: proc(
	n: int,
	store_inverse_twiddles, store_bitrev_table: bool,
	cand: Auto_C2C_Candidate,
	allocator: runtime.Allocator,
) -> (ok: bool, elapsed_ns: i64) {
	plan: C2C_Plan
	opts := C2C_Plan_Options{
		backend = cand.backend,
		store_inverse_twiddles = store_inverse_twiddles,
		store_bitrev_table = store_bitrev_table,
		threads = cand.threads,
		cooley_radix = cand.radix,
	}
	if err := c2c_plan_init_with_options(&plan, n, opts, allocator); err != .None {
		return false, 0
	}
	defer c2c_plan_destroy(&plan)

	data, data_err := make([]complex128, n, allocator)
	if data_err != .None {
		return false, 0
	}
	defer delete(data, allocator)

	autotune_fill_signal(data)
	if err := c2c_forward_in_place(&plan, data); err != .None {
		return false, 0
	}
	if err := c2c_inverse_in_place(&plan, data); err != .None {
		return false, 0
	}
	reps := autotune_reps_for_n(n)
	samples := autotune_samples_for_n(n)
	sample_times: [4]i64
	for sample_i in 0..<samples {
		// Warm one pair to reduce first-iteration bias inside each sample.
		if err := c2c_forward_in_place(&plan, data); err != .None {
			return false, 0
		}
		if err := c2c_inverse_in_place(&plan, data); err != .None {
			return false, 0
		}
		start := time.now()
		for _ in 0..<reps {
			if err := c2c_forward_in_place(&plan, data); err != .None {
				return false, 0
			}
			if err := c2c_inverse_in_place(&plan, data); err != .None {
				return false, 0
			}
		}
		elapsed := i64(time.duration_nanoseconds(time.since(start)))
		if elapsed <= 0 {
			elapsed = 1
		}
		sample_times[sample_i] = elapsed
	}
	stable_fast_elapsed := autotune_stable_fast_ns(sample_times[:samples])
	if stable_fast_elapsed <= 0 {
		stable_fast_elapsed = 1
	}
	return true, stable_fast_elapsed
}

resolve_auto_plan_options :: proc(n: int, options: C2C_Plan_Options, allocator: runtime.Allocator) -> C2C_Plan_Options {
	result := options
	if options.backend != .Auto {
		return result
	}

	if n < 1024 || n > (1 << 18) {
		return result
	}

	log2_n := log2_exact(n)
	base_threads := resolve_thread_count(options.threads, n)
	base_radix := resolve_cooley_radix(n, log2_n, options.cooley_radix, base_threads)
	allow_single_thread_candidates := base_threads <= 1 || n < (1 << 16)
	best := Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = base_radix, threads = base_threads}

	candidates: [8]Auto_C2C_Candidate
	candidate_count := 0
	auto_candidate_push_unique(&candidates, &candidate_count, best)
	if allow_single_thread_candidates {
		auto_candidate_push_unique(&candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 2, threads = 1})
	}
	if (log2_n & 1) == 0 {
		if allow_single_thread_candidates {
			auto_candidate_push_unique(&candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 4, threads = 1})
		}
		if base_threads > 1 {
			auto_candidate_push_unique(&candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 4, threads = base_threads})
		}
	}
	if base_threads > 1 {
		auto_candidate_push_unique(&candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 2, threads = base_threads})
	}
	if n <= 8192 {
		auto_candidate_push_unique(&candidates, &candidate_count, Auto_C2C_Candidate{backend = .Split_Radix, radix = 2, threads = 1})
	}

	best_elapsed := i64(-1)
	switch_margin := autotune_switch_margin_for_n(n)
	for i in 0..<candidate_count {
		cand := candidates[i]
		ok, elapsed := autotune_measure_candidate(n, options.store_inverse_twiddles, options.store_bitrev_table, cand, allocator)
		if !ok {
			continue
		}
		if best_elapsed < 0 {
			best_elapsed = elapsed
			best = cand
			continue
		}

		delta := f64(best_elapsed-elapsed) / f64(best_elapsed)
		if delta > switch_margin || (delta >= -switch_margin && autotune_prefer_candidate(n, cand, best)) {
			best_elapsed = elapsed
			best = cand
		}
	}

	result.backend = best.backend
	if options.cooley_radix == 0 {
		result.cooley_radix = best.radix
	}
	if options.threads <= 0 {
		result.threads = best.threads
	}
	return result
}

c2c_plan_estimate_bytes :: proc(n: int, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	if n < 1 {
		return 0
	}
	if !is_power_of_two(n) {
		if backend == .Split_Radix {
			return 0
		}
		m := next_power_of_two(2*n - 1)
		if m <= 0 {
			return 0
		}
		// bluestein_chirp + bluestein_work + bluestein_b_fft + convolution plan
		return n*size_of(complex128) + 2*m*size_of(complex128) +
			c2c_plan_estimate_bytes(m, .Cooley_Tukey, store_inverse_twiddles, store_bitrev_table)
	}
	resolved_backend := resolve_backend_for_size(n, backend, context.allocator)
	bytes := 0

	bytes += (n / 2) * size_of(complex128)
	if resolved_backend == .Cooley_Tukey && store_inverse_twiddles {
		bytes += (n / 2) * size_of(complex128)
	}
	switch resolved_backend {
	case .Cooley_Tukey:
		if store_bitrev_table {
			bytes += n * size_of(u32)
		}
	case .Split_Radix:
		bytes += n * size_of(complex128)
	case .Auto:
		if store_bitrev_table {
			bytes += n * size_of(u32)
		}
	}
	return bytes
}

resolve_backend_for_size :: proc(n: int, requested: Backend, allocator: runtime.Allocator) -> Backend {
	_ = allocator
	if requested != .Auto {
		return requested
	}
	// Fast heuristic with deterministic behavior:
	// Cooley-Tukey is generally faster in this implementation.
	// Keep Split-Radix auto-pick only as a single-core fallback at N=65536.
	if n == 65536 {
		if os.get_processor_core_count() <= 1 {
			return .Split_Radix
		}
		return .Cooley_Tukey
	}
	return .Cooley_Tukey
}

c2c_plan_destroy :: proc(plan: ^C2C_Plan) {
	cooley_tukey_parallel_shutdown(plan)
	cooley_tukey_radix2_simd_destroy(plan)
	cooley_tukey_radix4_simd_destroy(plan)
	if plan.bluestein_conv_plan != nil {
		c2c_plan_destroy(plan.bluestein_conv_plan)
		free(plan.bluestein_conv_plan, plan.allocator)
	}
	if plan.bluestein_m != nil {
		delete(plan.bluestein_m, plan.allocator)
	}
	if plan.bluestein_chirp != nil {
		delete(plan.bluestein_chirp, plan.allocator)
	}
	if plan.bluestein_b_fft != nil {
		delete(plan.bluestein_b_fft, plan.allocator)
	}
	if plan.bitrev != nil {
		delete(plan.bitrev, plan.allocator)
	}
	if plan.digitrev4 != nil {
		delete(plan.digitrev4, plan.allocator)
	}
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

r2c_plan_destroy :: proc(plan: ^R2C_Plan) {
	c2c_plan_destroy(&plan.c2c)
	if plan.twiddles != nil {
		delete(plan.twiddles, plan.allocator)
	}
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	plan^ = {}
}

c2c_plan_2d_destroy :: proc(plan: ^C2C_Plan_2D) {
	c2c_plan_destroy(&plan.row_plan)
	c2c_plan_destroy(&plan.col_plan)
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	plan^ = {}
}

c2c_plan_init_with_backend :: proc(plan: ^C2C_Plan, n: int, backend: Backend, allocator := context.allocator) -> Error {
	opts := C2C_Plan_Options{
		backend = backend,
		store_inverse_twiddles = true,
		store_bitrev_table = true,
		threads = 0,
		cooley_radix = 0,
	}
	return c2c_plan_init_with_options(plan, n, opts, allocator)
}

c2c_plan_init_with_options :: proc(plan: ^C2C_Plan, n: int, options: C2C_Plan_Options, allocator := context.allocator) -> Error {
	if n < 1 {
		return .Invalid_Length
	}
	if !is_power_of_two(n) {
		adjusted := options
		// Split-Radix only supports power-of-two. Non-power-of-two falls back
		// to Bluestein + Cooley-Tukey convolution.
		if adjusted.backend == .Split_Radix {
			adjusted.backend = .Cooley_Tukey
		}
		return c2c_plan_init_bluestein(plan, n, adjusted, allocator)
	}
	effective_options := resolve_auto_plan_options(n, options, allocator)
	c2c_plan_destroy(plan)
	resolved_backend := resolve_backend_for_size(n, effective_options.backend, allocator)

	twiddles, twiddle_err := make([]complex128, n/2, allocator)
	if twiddle_err != .None {
		return .Allocation_Failed
	}
	should_store_inverse_twiddles := resolved_backend == .Cooley_Tukey && effective_options.store_inverse_twiddles
	should_store_bitrev_table := resolved_backend == .Cooley_Tukey && effective_options.store_bitrev_table
	twiddles_inv: []complex128
	if should_store_inverse_twiddles {
		inv_alloc_err: runtime.Allocator_Error
		twiddles_inv, inv_alloc_err = make([]complex128, n/2, allocator)
		if inv_alloc_err != .None {
			delete(twiddles, allocator)
			return .Allocation_Failed
		}
	}

	#no_bounds_check for k in 0..<len(twiddles) {
		angle := -2.0 * math.PI * f64(k) / f64(n)
		s, c := math.sincos(angle)
		w := complex(c, s)
		twiddles[k] = w
		if should_store_inverse_twiddles {
			twiddles_inv[k] = conj(w)
		}
	}

	plan.n = n
	plan.log2_n = log2_exact(n)
	plan.backend = resolved_backend
	plan.uses_bluestein = false
	plan.store_inverse_twiddles = should_store_inverse_twiddles
	plan.store_bitrev_table = should_store_bitrev_table
	plan.num_threads = resolve_thread_count(effective_options.threads, n)
	plan.cooley_radix = 2
	plan.twiddles = twiddles
	plan.twiddles_inv = twiddles_inv
	plan.allocator = allocator

	switch resolved_backend {
	case .Cooley_Tukey:
		plan.cooley_radix = resolve_cooley_radix(n, plan.log2_n, effective_options.cooley_radix, plan.num_threads)
		if should_store_bitrev_table {
			if plan.cooley_radix == 4 {
				digit_count := plan.log2_n / 2
				digitrev4, dr_err := make([]u32, n, allocator)
				if dr_err != .None {
					c2c_plan_destroy(plan)
					return .Allocation_Failed
				}
				#no_bounds_check for i in 0..<n {
					digitrev4[i] = reverse_base4_u32(u32(i), digit_count)
				}
				plan.digitrev4 = digitrev4
			} else {
				bitrev, bitrev_err := make([]u32, n, allocator)
				if bitrev_err != .None {
					c2c_plan_destroy(plan)
					return .Allocation_Failed
				}
				#no_bounds_check for i in 0..<n {
					bitrev[i] = reverse_bits_u32(u32(i), plan.log2_n)
				}
				plan.bitrev = bitrev
			}
		}
	case .Split_Radix:
		plan.num_threads = 1
		plan.cooley_radix = 2
		scratch, scratch_err := make([]complex128, n, allocator)
		if scratch_err != .None {
			c2c_plan_destroy(plan)
			return .Allocation_Failed
		}
		plan.scratch = scratch
	case .Auto:
		plan.num_threads = 1
		plan.cooley_radix = 2
		bitrev, bitrev_err := make([]u32, n, allocator)
		if bitrev_err != .None {
			c2c_plan_destroy(plan)
			return .Allocation_Failed
		}
		#no_bounds_check for i in 0..<n {
			bitrev[i] = reverse_bits_u32(u32(i), plan.log2_n)
		}
		plan.bitrev = bitrev
	}

	if resolved_backend == .Cooley_Tukey && plan.cooley_radix == 2 {
		if err := cooley_tukey_radix2_simd_init(plan); err != .None {
			c2c_plan_destroy(plan)
			return err
		}
		if err := cooley_tukey_parallel_init(plan); err != .None {
			c2c_plan_destroy(plan)
			return err
		}
	} else if resolved_backend == .Cooley_Tukey && plan.cooley_radix == 4 {
		if should_store_bitrev_table {
			if err := cooley_tukey_radix4_simd_init(plan); err != .None {
				c2c_plan_destroy(plan)
				return err
			}
		}
	}

	plan.initialized = true
	return .None
}

c2c_plan_init :: proc(plan: ^C2C_Plan, n: int, allocator := context.allocator) -> Error {
	return c2c_plan_init_with_backend(plan, n, .Auto, allocator)
}

c2c_plan_init_low_ram :: proc(plan: ^C2C_Plan, n: int, backend := Backend.Auto, allocator := context.allocator) -> Error {
	opts := C2C_Plan_Options{
		backend = backend,
		store_inverse_twiddles = false,
		store_bitrev_table = false,
		threads = 0,
		cooley_radix = 0,
	}
	return c2c_plan_init_with_options(plan, n, opts, allocator)
}

r2c_plan_init_with_backend :: proc(plan: ^R2C_Plan, n: int, backend: Backend, allocator := context.allocator) -> Error {
	if n < 2 {
		return .Invalid_Length
	}
	r2c_plan_destroy(plan)

	half_n := n / 2
	if is_power_of_two(n) {
		c2c_err := c2c_plan_init_with_backend(&plan.c2c, half_n, backend, allocator)
		if c2c_err != .None {
			return c2c_err
		}

		twiddles, twiddle_err := make([]complex128, half_n+1, allocator)
		if twiddle_err != .None {
			r2c_plan_destroy(plan)
			return .Allocation_Failed
		}
		scratch, scratch_err := make([]complex128, half_n, allocator)
		if scratch_err != .None {
			delete(twiddles, allocator)
			r2c_plan_destroy(plan)
			return .Allocation_Failed
		}

		#no_bounds_check for k in 0..<len(twiddles) {
			angle := -2.0 * math.PI * f64(k) / f64(n)
			s, c := math.sincos(angle)
			twiddles[k] = complex(c, s)
		}

		plan.twiddles = twiddles
		plan.scratch = scratch
		plan.uses_full_c2c = false
	} else {
		c2c_err := c2c_plan_init_with_backend(&plan.c2c, n, backend, allocator)
		if c2c_err != .None {
			return c2c_err
		}

		scratch, scratch_err := make([]complex128, n, allocator)
		if scratch_err != .None {
			r2c_plan_destroy(plan)
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

r2c_plan_init :: proc(plan: ^R2C_Plan, n: int, allocator := context.allocator) -> Error {
	return r2c_plan_init_with_backend(plan, n, .Auto, allocator)
}

c2c_plan_2d_init_with_backend :: proc(plan: ^C2C_Plan_2D, rows, cols: int, backend: Backend, allocator := context.allocator) -> Error {
	if rows < 1 || cols < 1 {
		return .Invalid_Length
	}
	c2c_plan_2d_destroy(plan)

	row_err := c2c_plan_init_with_backend(&plan.row_plan, cols, backend, allocator)
	if row_err != .None {
		return row_err
	}
	col_err := c2c_plan_init_with_backend(&plan.col_plan, rows, backend, allocator)
	if col_err != .None {
		c2c_plan_destroy(&plan.row_plan)
		return col_err
	}

	scratch, sc_err := make([]complex128, rows, allocator)
	if sc_err != .None {
		c2c_plan_destroy(&plan.row_plan)
		c2c_plan_destroy(&plan.col_plan)
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

c2c_plan_2d_init :: proc(plan: ^C2C_Plan_2D, rows, cols: int, allocator := context.allocator) -> Error {
	return c2c_plan_2d_init_with_backend(plan, rows, cols, .Auto, allocator)
}

c2c_forward_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.n {
		return .Size_Mismatch
	}
	if len(data) <= 1 {
		return .None
	}
	if plan.uses_bluestein {
		return bluestein_forward_in_place(plan, data)
	}

	switch plan.backend {
	case .Cooley_Tukey:
		if plan.cooley_radix == 4 {
			return cooley_tukey_forward_radix4_in_place(plan, data)
		}
		return cooley_tukey_forward_in_place(plan, data)
	case .Split_Radix:
		return split_radix_forward_in_place(plan, data)
	case .Auto:
		return cooley_tukey_forward_in_place(plan, data)
	}

	return .None
}

c2c_inverse_in_place :: proc(plan: ^C2C_Plan, data: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.n {
		return .Size_Mismatch
	}
	if len(data) <= 1 {
		return .None
	}
	if plan.uses_bluestein {
		return bluestein_inverse_in_place(plan, data)
	}

	switch plan.backend {
	case .Cooley_Tukey:
		if plan.cooley_radix == 4 {
			return cooley_tukey_inverse_radix4_in_place(plan, data)
		}
		return cooley_tukey_inverse_in_place(plan, data)
	case .Split_Radix:
		return split_radix_inverse_in_place(plan, data)
	case .Auto:
		return cooley_tukey_inverse_in_place(plan, data)
	}

	return .None
}

c2c_forward :: proc(plan: ^C2C_Plan, input: []complex128, output: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.n {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_forward_in_place(plan, output)
}

c2c_inverse :: proc(plan: ^C2C_Plan, input: []complex128, output: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.n {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_inverse_in_place(plan, output)
}

r2c_forward_with_scratch :: proc(plan: ^R2C_Plan, input: []f64, output: []complex128, scratch: []complex128) -> Error {
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
			scratch[i] = complex(input[i], 0.0)
		}
		if err := c2c_forward_in_place(&plan.c2c, scratch); err != .None {
			return err
		}
		copy(output, scratch[:plan.half_n+1])
		return .None
	}
	if len(scratch) != plan.half_n {
		return .Size_Mismatch
	}

	#no_bounds_check for i in 0..<plan.half_n {
		re := input[2*i]
		im := input[2*i + 1]
		scratch[i] = complex(re, im)
	}

	if err := c2c_forward_in_place(&plan.c2c, scratch); err != .None {
		return err
	}

	a0 := scratch[0]
	output[0] = complex(real(a0)+imag(a0), 0.0)
	output[plan.half_n] = complex(real(a0)-imag(a0), 0.0)

	#no_bounds_check for k in 1..<plan.half_n {
		a := scratch[k]
		b := conj(scratch[plan.half_n-k])

		t1 := 0.5 * (a + b)
		t2 := complex(0.0, -0.5) * (a - b)
		output[k] = t1 + plan.twiddles[k]*t2
	}

	return .None
}

c2r_inverse_with_scratch :: proc(plan: ^R2C_Plan, input: []complex128, output: []f64, scratch: []complex128) -> Error {
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
			scratch[i] = 0
		}
		#no_bounds_check for k in 0..=plan.half_n {
			scratch[k] = input[k]
		}
		#no_bounds_check for k in 1..=plan.half_n {
			mirror := plan.n - k
			if mirror != k {
				scratch[mirror] = conj(input[k])
			} else {
				scratch[k] = complex(real(input[k]), 0.0)
			}
		}

		if err := c2c_inverse_in_place(&plan.c2c, scratch); err != .None {
			return err
		}
		#no_bounds_check for i in 0..<plan.n {
			output[i] = real(scratch[i])
		}
		return .None
	}
	if len(scratch) != plan.half_n {
		return .Size_Mismatch
	}

	x0 := input[0]
	xn2 := input[plan.half_n]
	scratch[0] = complex(
		0.5*(real(x0)+real(xn2)),
		0.5*(real(x0)-real(xn2)),
	)

	#no_bounds_check for k in 1..<plan.half_n {
		c := input[k]
		d := conj(input[plan.half_n-k])
		sum := c + d
		diff := c - d
		wc := conj(plan.twiddles[k])
		scratch[k] = 0.5 * (sum + complex(0.0, 1.0)*diff*wc)
	}

	if err := c2c_inverse_in_place(&plan.c2c, scratch); err != .None {
		return err
	}

	#no_bounds_check for i in 0..<plan.half_n {
		v := scratch[i]
		output[2*i] = real(v)
		output[2*i+1] = imag(v)
	}

	return .None
}

r2c_forward :: proc(plan: ^R2C_Plan, input: []f64, output: []complex128) -> Error {
	return r2c_forward_with_scratch(plan, input, output, plan.scratch)
}

c2r_inverse :: proc(plan: ^R2C_Plan, input: []complex128, output: []f64) -> Error {
	return c2r_inverse_with_scratch(plan, input, output, plan.scratch)
}

c2c_2d_forward_in_place_with_scratch :: proc(plan: ^C2C_Plan_2D, data: []complex128, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.rows*plan.cols || len(scratch) != plan.rows {
		return .Size_Mismatch
	}

	for r in 0..<plan.rows {
		row := data[r*plan.cols:][:plan.cols]
		if err := c2c_forward_in_place(&plan.row_plan, row); err != .None {
			return err
		}
	}

	for c in 0..<plan.cols {
		#no_bounds_check for r in 0..<plan.rows {
			scratch[r] = data[r*plan.cols+c]
		}
		if err := c2c_forward_in_place(&plan.col_plan, scratch[:plan.rows]); err != .None {
			return err
		}
		#no_bounds_check for r in 0..<plan.rows {
			data[r*plan.cols+c] = scratch[r]
		}
	}

	return .None
}

c2c_2d_inverse_in_place_with_scratch :: proc(plan: ^C2C_Plan_2D, data: []complex128, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.rows*plan.cols || len(scratch) != plan.rows {
		return .Size_Mismatch
	}

	for r in 0..<plan.rows {
		row := data[r*plan.cols:][:plan.cols]
		if err := c2c_inverse_in_place(&plan.row_plan, row); err != .None {
			return err
		}
	}

	for c in 0..<plan.cols {
		#no_bounds_check for r in 0..<plan.rows {
			scratch[r] = data[r*plan.cols+c]
		}
		if err := c2c_inverse_in_place(&plan.col_plan, scratch[:plan.rows]); err != .None {
			return err
		}
		#no_bounds_check for r in 0..<plan.rows {
			data[r*plan.cols+c] = scratch[r]
		}
	}

	return .None
}

c2c_2d_forward_in_place :: proc(plan: ^C2C_Plan_2D, data: []complex128) -> Error {
	return c2c_2d_forward_in_place_with_scratch(plan, data, plan.scratch)
}

c2c_2d_inverse_in_place :: proc(plan: ^C2C_Plan_2D, data: []complex128) -> Error {
	return c2c_2d_inverse_in_place_with_scratch(plan, data, plan.scratch)
}

c2c_2d_forward :: proc(plan: ^C2C_Plan_2D, input: []complex128, output: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_2d_forward_in_place(plan, output)
}

c2c_2d_inverse :: proc(plan: ^C2C_Plan_2D, input: []complex128, output: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	copy(output, input)
	return c2c_2d_inverse_in_place(plan, output)
}
