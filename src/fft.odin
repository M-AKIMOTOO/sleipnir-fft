package fft

import "base:runtime"
import "base:intrinsics"
import "core:math"
import "core:os"
import "core:simd"
import sysinfo "core:sys/info"
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
	cooley_radix:          int,  // 0=auto, 2=radix-2, 4=radix-4, 8=radix-8, 16=radix-16
}

C2C_Parallel_Mode :: enum {
	Idle,
	Forward,
	Forward_R4,
	Forward_R4_Chunk,
	Inverse_Inv,
	Inverse_R4_Inv,
	Inverse_R4_Inv_Chunk,
	Inverse_Conj,
	Inverse_R4_Conj,
	Inverse_R4_Conj_Chunk,
	Stop,
}

C2C_2D_Parallel_Mode :: enum {
	Idle,
	Rows_Forward,
	Rows_Inverse,
	Cols_Forward,
	Cols_Inverse,
	Stop,
}

R2R_2D_Parallel_Mode :: enum {
	Idle,
	Rows_Forward,
	Rows_Inverse,
	Cols_Forward,
	Cols_Inverse,
	Stop,
}

FFT_2D_Axis_Transform :: enum {
	C2C_Forward,
	C2C_Inverse,
	R2C_Forward,
	C2R_Inverse,
}

FFT_2D_Pass_Order :: enum {
	Row_Then_Col,
	Col_Then_Row,
}

FFT_2D_Exec_Descriptor :: struct {
	row_kind:     FFT_2D_Axis_Transform,
	col_kind:     FFT_2D_Axis_Transform,
	pass_order:   FFT_2D_Pass_Order,
	rows:         int,
	cols:         int,
	freq_cols:    int,
	c2c_plan:     ^C2C_Plan,
	r2c_plan:     ^R2C_Plan,
	col_plan:     ^C2C_Plan,
	complex_data: []complex128,
	scratch:      []complex128,
	freq_scratch: []complex128,
	real_input:   []f64,
	real_output:  []f64,
}

C2C_Parallel_Worker_Context :: struct {
	plan:         ^C2C_Plan,
	worker_index: int,
}

C2C_2D_Parallel_Worker_Context :: struct {
	plan:         ^C2C_Plan_2D,
	worker_index: int,
}

R2R_2D_Parallel_Worker_Context :: struct {
	plan:         ^R2R_Plan_2D,
	worker_index: int,
}

Radix4_Twiddle_Pair_Pack :: struct {
	w0_re: f64,
	w0_im: f64,
	w1_re: f64,
	w1_im: f64,
}

Radix4_Twiddle_Triple_Pack :: struct {
	w1: Radix4_Twiddle_Pair_Pack,
	w2: Radix4_Twiddle_Pair_Pack,
	w3: Radix4_Twiddle_Pair_Pack,
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
	tw: []Radix4_Twiddle_Triple_Pack,
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
	twiddles_inv: []complex128,
	scratch:     []complex128,
	allocator:   runtime.Allocator,
	initialized: bool,
}

R2R_Kind :: enum {
	DCT_II,
}

R2R_Plan :: struct {
	n:           int,
	kind:        R2R_Kind,
	uses_full_c2c: bool,
	r2c:         R2C_Plan,
	c2c:         C2C_Plan,
	twiddles:    []complex128,
	scratch:     []f64,
	freq_scratch: []complex128,
	complex_scratch: []complex128,
	allocator:   runtime.Allocator,
	initialized: bool,
}

R2C_Plan_2D :: struct {
	rows:        int,
	cols:        int,
	freq_cols:   int,
	backend:     Backend,
	row_plan:    R2C_Plan,
	col_plan:    C2C_Plan,
	scratch:     []complex128,
	freq_scratch: []complex128,
	allocator:   runtime.Allocator,
	initialized: bool,
}

R2R_Plan_2D :: struct {
	rows:        int,
	cols:        int,
	kind:        R2R_Kind,
	backend:     Backend,
	row_plan:    R2R_Plan,
	col_plan:    R2R_Plan,
	scratch:     []f64,
	parallel_enabled: bool,
	parallel_worker_count: int,
	parallel_threads: []^thread.Thread,
	parallel_ctx: []R2R_2D_Parallel_Worker_Context,
	parallel_start: sync.Barrier,
	parallel_done: sync.Barrier,
	parallel_mode: R2R_2D_Parallel_Mode,
	parallel_data: []f64,
	parallel_block_scratch: []f64,
	parallel_real_scratch: []f64,
	parallel_freq_scratch: []complex128,
	parallel_complex_scratch: []complex128,
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
	parallel_enabled: bool,
	parallel_worker_count: int,
	parallel_threads: []^thread.Thread,
	parallel_ctx: []C2C_2D_Parallel_Worker_Context,
	parallel_start: sync.Barrier,
	parallel_done: sync.Barrier,
	parallel_mode: C2C_2D_Parallel_Mode,
	parallel_data: []complex128,
	parallel_scratch: []complex128,
	allocator:   runtime.Allocator,
	initialized: bool,
}

C2C_Plan_2D_F32 :: struct {
	rows:        int,
	cols:        int,
	backend:     Backend,
	row_plan:    C2C_Plan_F32,
	col_plan:    C2C_Plan_F32,
	scratch:     []complex64,
	allocator:   runtime.Allocator,
	initialized: bool,
}

C2C_2D_BLOCK_MIN_POINTS :: 1 << 17
C2C_2D_TARGET_SCRATCH_BYTES :: 1 << 18
C2C_2D_MAX_SCRATCH_COLS :: 16
C2C_2D_LARGE_MAX_SCRATCH_COLS :: 32
C2C_2D_PARALLEL_MIN_POINTS :: 1 << 18
C2C_2D_PARALLEL_MAX_WORKERS :: 8

FFT_USE_AVX2 :: ODIN_ARCH == .amd64 && intrinsics.has_target_feature("avx2")
FFT_USE_ARM_SIMD :: ODIN_ARCH == .arm64 && intrinsics.has_target_feature("neon")
FFT_USE_SIMD_KERNELS :: FFT_USE_AVX2 || FFT_USE_ARM_SIMD

SIMD_FWD_T3_SIGN :: simd.f64x4{1.0, -1.0, 1.0, -1.0}
SIMD_INV_T3_SIGN :: simd.f64x4{-1.0, 1.0, -1.0, 1.0}
SIMD_CONJ_SIGN   :: simd.f64x4{1.0, -1.0, 1.0, -1.0}

transpose_4x4_complex_block :: #force_inline proc(
	r0, r1, r2, r3: [^]complex128, // Input rows
	c0, c1, c2, c3: [^]complex128, // Output columns
) {
	c0[0], c0[1], c0[2], c0[3] = r0[0], r1[0], r2[0], r3[0]
	c1[0], c1[1], c1[2], c1[3] = r0[1], r1[1], r2[1], r3[1]
	c2[0], c2[1], c2[2], c2[3] = r0[2], r1[2], r2[2], r3[2]
	c3[0], c3[1], c3[2], c3[3] = r0[3], r1[3], r2[3], r3[3]
}

scale_complex_array_in_place :: #force_inline proc(data: []complex128, scale: f64) {
	n := len(data)
	if n == 0 { return }
	when FFT_USE_SIMD_KERNELS {
		i := 0
		v_scale := simd.f64x4{scale, scale, scale, scale}
		for ; i + 1 < n; i += 2 {
			v := intrinsics.unaligned_load((^simd.f64x4)(&data[i]))
			res := v * v_scale
			intrinsics.unaligned_store((^simd.f64x4)(&data[i]), res)
		}
		for ; i < n; i += 1 {
			data[i] *= complex(scale, 0.0)
		}
	} else {
		#no_bounds_check for i in 0..<n {
			data[i] *= complex(scale, 0.0)
		}
	}
}

scale_copy_complex_array :: #force_inline proc(dst, src: []complex128, scale: f64) {
	n := len(dst)
	if n == 0 { return }
	when FFT_USE_SIMD_KERNELS {
		i := 0
		v_scale := simd.f64x4{scale, scale, scale, scale}
		for ; i + 1 < n; i += 2 {
			v := intrinsics.unaligned_load((^simd.f64x4)(&src[i]))
			res := v * v_scale
			intrinsics.unaligned_store((^simd.f64x4)(&dst[i]), res)
		}
		for ; i < n; i += 1 {
			dst[i] = src[i] * complex(scale, 0.0)
		}
	} else {
		#no_bounds_check for i in 0..<n {
			dst[i] = src[i] * complex(scale, 0.0)
		}
	}
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

r2c_2d_output_cols :: #force_inline proc(cols: int) -> int {
	return r2c_output_len(cols)
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
	profile := current_cpu_profile()
	core_count := profile.physical_cores
	if core_count <= 1 {
		return 1
	}
	if n >= (1 << 20) {
		return min(core_count, 16)
	}
	if n >= (1 << 18) {
		return min(core_count, 8)
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
	if requested == 8 {
		if n >= 8 {
			return 8
		}
		return 2
	}
	if requested == 16 {
		if n >= 16 {
			return 16
		}
		if n >= 8 {
			return 8
		}
		return 2
	}
	// Auto heuristic:
	profile := current_cpu_profile()
	if n <= 4 {
		return 2
	}
	if !profile.supports_avx2 {
		if (log2_n & 1) == 0 && n >= 16 {
			return 4
		}
		return 2
	}
	if num_threads > 1 && n >= (1 << 18) {
		if (log2_n & 1) == 0 {
			return 4
		}
		return 2
	}
	switch profile.tier {
	case .AVX512_Class:
		if n >= 256 && (log2_n & 1) == 0 {
			return 4
		}
	case .AVX2_FMA_Class:
		if n >= 64 && (log2_n & 1) == 0 {
			return 4
		}
	case .AVX2_Class:
		if n >= 256 && (log2_n & 1) == 0 {
			return 4
		}
	case .SSE_Class, .Scalar, .NEON_Class:
	}
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

AUTO_C2C_TUNE_SINGLE_MAX_N :: 1 << 20
AUTO_C2C_TUNE_MULTI_MAX_N  :: 1 << 13
AUTO_C2C_CACHE_CAPACITY    :: 32
AUTO_R2C_TUNE_MIN_N        :: 1 << 15
AUTO_R2C_TUNE_MAX_N        :: 1 << 20
AUTO_R2C_CACHE_CAPACITY    :: 16

Auto_C2C_Cache_Key :: struct {
	n:                     int,
	threads:               int,
	requested_cooley_radix: int,
	store_inverse_twiddles: bool,
	store_bitrev_table:    bool,
}

Auto_C2C_Cache_Entry :: struct {
	valid:   bool,
	key:     Auto_C2C_Cache_Key,
	best:    Auto_C2C_Candidate,
}

auto_c2c_cache: [AUTO_C2C_CACHE_CAPACITY]Auto_C2C_Cache_Entry
auto_c2c_cache_next: int

Auto_R2C_Cache_Entry :: struct {
	valid: bool,
	n:     int,
	best:  Backend,
}

CPU_Tier :: enum {
	Scalar,
	SSE_Class,
	NEON_Class,
	AVX2_Class,
	AVX2_FMA_Class,
	AVX512_Class,
}

CPU_Profile :: struct {
	tier:            CPU_Tier,
	physical_cores:  int,
	logical_cores:   int,
	supports_avx2:   bool,
	supports_fma:    bool,
	supports_avx512: bool,
	supports_neon:   bool,
}

auto_r2c_cache: [AUTO_R2C_CACHE_CAPACITY]Auto_R2C_Cache_Entry
auto_r2c_cache_next: int
cpu_profile_cached: CPU_Profile
cpu_profile_initialized: bool

current_cpu_profile :: proc() -> CPU_Profile {
	if cpu_profile_initialized {
		return cpu_profile_cached
	}

	profile := CPU_Profile{
		tier = .Scalar,
		physical_cores = 1,
		logical_cores = os.get_processor_core_count(),
	}
	if profile.logical_cores < 1 {
		profile.logical_cores = 1
	}
	if physical, logical, ok := sysinfo.cpu_core_count(); ok {
		if physical > 0 {
			profile.physical_cores = physical
		}
		if logical > 0 {
			profile.logical_cores = logical
		}
	}
	if profile.physical_cores < 1 {
		profile.physical_cores = max(1, profile.logical_cores / 2)
	}

	when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
		features := sysinfo.cpu_features()
		profile.supports_avx2 = .avx2 in features
		profile.supports_fma = .fma in features
		profile.supports_avx512 = .avx512f in features && .avx512vl in features
		switch {
		case profile.supports_avx512:
			profile.tier = .AVX512_Class
		case profile.supports_avx2 && profile.supports_fma:
			profile.tier = .AVX2_FMA_Class
		case profile.supports_avx2:
			profile.tier = .AVX2_Class
		case .sse2 in features:
			profile.tier = .SSE_Class
		case:
			profile.tier = .Scalar
		}
	} else when ODIN_ARCH == .arm64 {
		// Advanced SIMD (NEON) is mandatory for AArch64, including Windows ARM64
		// where the OS CPU-feature query may not expose an equivalent flag.
		profile.supports_neon = true
		profile.tier = .NEON_Class
	} else when ODIN_ARCH == .arm32 {
		features := sysinfo.cpu_features()
		profile.supports_neon = .asimd in features
		if profile.supports_neon {
			profile.tier = .NEON_Class
		}
	} else {
		profile.tier = .Scalar
	}

	cpu_profile_cached = profile
	cpu_profile_initialized = true
	return profile
}

autotune_reps_for_n :: #force_inline proc(n: int) -> int {
	if n <= 1 << 10 {
		return 16
	}
	if n <= 1 << 12 {
		return 12
	}
	if n <= 1 << 14 {
		return 10
	}
	if n <= 1 << 16 {
		return 8
	}
	if n <= 1 << 17 {
		return 4
	}
	if n <= 1 << 18 {
		return 3
	}
	if n <= 1 << 19 {
		return 2
	}
	return 1
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

autotune_r2c_inverse_reps_for_n :: #force_inline proc(n: int) -> int {
	return max(3, autotune_reps_for_n(n) * 2)
}

resolve_small_c2c_single_candidate :: #force_inline proc(n: int) -> (Auto_C2C_Candidate, bool) {
	switch n {
	case 2:
		return Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 2, threads = 1}, true
	case 4, 8, 32, 128, 512:
		return Auto_C2C_Candidate{backend = .Split_Radix, radix = 2, threads = 1}, true
	case 16, 64, 256:
		return Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 4, threads = 1}, true
	}
	return Auto_C2C_Candidate{}, false
}

autotune_r2c_inverse_samples_for_n :: #force_inline proc(n: int) -> int {
	return min(4, autotune_samples_for_n(n) + 1)
}

autotune_switch_margin_for_n :: #force_inline proc(n: int) -> f64 {
	// Require a minimum relative win before switching to avoid noise-driven flips.
	if n <= 1 << 14 {
		return 0.02
	}
	if n <= 1 << 16 {
		return 0.05
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
	if cand.threads <= 1 {
		if (n == 1024 || n == 4096) && cand.backend == .Cooley_Tukey && cand.radix == 4 {
			// Keep autotune stable on power-of-four sizes where radix-4
			// reliably wins but the gap can be hidden by short-sample noise.
			score -= 3
		}
		if (n == 2048 || n == 8192) && cand.backend == .Split_Radix {
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

auto_candidate_canonicalize :: #force_inline proc(n: int, cand: Auto_C2C_Candidate) -> Auto_C2C_Candidate {
	result := cand
	if cand.backend == .Cooley_Tukey {
		log2_n := log2_exact(n)
		result.radix = resolve_cooley_radix(n, log2_n, cand.radix, cand.threads)
	} else {
		result.radix = 2
	}
	if cand.threads < 1 {
		result.threads = 1
	}
	return result
}

auto_candidate_equal :: #force_inline proc(a, b: Auto_C2C_Candidate) -> bool {
	return a.backend == b.backend && a.radix == b.radix && a.threads == b.threads
}

auto_c2c_cache_key_equal :: #force_inline proc(a, b: Auto_C2C_Cache_Key) -> bool {
	return a.n == b.n &&
		a.threads == b.threads &&
		a.requested_cooley_radix == b.requested_cooley_radix &&
		a.store_inverse_twiddles == b.store_inverse_twiddles &&
		a.store_bitrev_table == b.store_bitrev_table
}

auto_c2c_cache_lookup :: proc(key: Auto_C2C_Cache_Key) -> (Auto_C2C_Candidate, bool) {
	#no_bounds_check for i in 0..<len(auto_c2c_cache) {
		entry := auto_c2c_cache[i]
		if entry.valid && auto_c2c_cache_key_equal(entry.key, key) {
			return entry.best, true
		}
	}
	return Auto_C2C_Candidate{}, false
}

auto_c2c_cache_store :: proc(key: Auto_C2C_Cache_Key, best: Auto_C2C_Candidate) {
	index := auto_c2c_cache_next % len(auto_c2c_cache)
	auto_c2c_cache[index] = Auto_C2C_Cache_Entry{
		valid = true,
		key = key,
		best = best,
	}
	auto_c2c_cache_next += 1
}

auto_candidate_push_unique :: #force_inline proc(n: int, candidates: ^[8]Auto_C2C_Candidate, candidate_count: ^int, cand: Auto_C2C_Candidate) {
	candidate := auto_candidate_canonicalize(n, cand)
	count := candidate_count^
	#no_bounds_check for i in 0..<count {
		if auto_candidate_equal(candidates[i], candidate) {
			return
		}
	}
	if count >= len(candidates^) {
		return
	}
	candidates[count] = candidate
	candidate_count^ = count + 1
}

autotune_fill_signal :: proc(data: []complex128) {
	#no_bounds_check for i in 0..<len(data) {
		re := f64(i & 31) * 0.03125
		im := f64((i * 3) & 31) * 0.03125
		data[i] = complex(re, im)
	}
}

autotune_fill_real_signal :: proc(data: []f64) {
	#no_bounds_check for i in 0..<len(data) {
		data[i] = 0.5*math.sin(0.017*f64(i)) + 0.25*math.cos(0.031*f64((i*3)&1023))
	}
}

copy_complex_to_interleaved_real :: proc(dst: []f64, src: []complex128) {
	when FFT_USE_SIMD_KERNELS {
		i := 0
		for ; i+1 < len(src); i += 2 {
			v := intrinsics.unaligned_load(cast(^simd.f64x4)(&src[i]))
			intrinsics.unaligned_store(cast(^simd.f64x4)(&dst[2*i]), v)
		}
		for ; i < len(src); i += 1 {
			v := src[i]
			dst[2*i] = real(v)
			dst[2*i+1] = imag(v)
		}
		return
	}

	#no_bounds_check for i in 0..<len(src) {
		v := src[i]
		dst[2*i] = real(v)
		dst[2*i+1] = imag(v)
	}
}

auto_r2c_cache_lookup :: proc(n: int) -> (Backend, bool) {
	#no_bounds_check for i in 0..<len(auto_r2c_cache) {
		entry := auto_r2c_cache[i]
		if entry.valid && entry.n == n {
			return entry.best, true
		}
	}
	return .Auto, false
}

auto_r2c_cache_store :: proc(n: int, best: Backend) {
	index := auto_r2c_cache_next % len(auto_r2c_cache)
	auto_r2c_cache[index] = Auto_R2C_Cache_Entry{
		valid = true,
		n = n,
		best = best,
	}
	auto_r2c_cache_next += 1
}

autotune_measure_r2c_backend_inverse :: proc(n: int, backend: Backend, allocator: runtime.Allocator) -> (ok: bool, elapsed_ns: i64) {
	plan: R2C_Plan
	if err := r2c_plan_init_with_backend(&plan, n, backend, allocator); err != .None {
		return false, 0
	}
	defer r2c_plan_destroy(&plan)

	input, input_err := make([]f64, n, allocator)
	if input_err != .None {
		return false, 0
	}
	defer delete(input, allocator)

	output, output_err := make([]complex128, r2c_output_len(n), allocator)
	if output_err != .None {
		return false, 0
	}
	defer delete(output, allocator)

	restore, restore_err := make([]f64, n, allocator)
	if restore_err != .None {
		return false, 0
	}
	defer delete(restore, allocator)

	autotune_fill_real_signal(input)
	if err := r2c_forward(&plan, input, output); err != .None {
		return false, 0
	}
	if err := c2r_inverse(&plan, output, restore); err != .None {
		return false, 0
	}

	reps := autotune_r2c_inverse_reps_for_n(n)
	samples := autotune_r2c_inverse_samples_for_n(n)
	sample_times: [4]i64
	for sample_i in 0..<samples {
		if err := c2r_inverse(&plan, output, restore); err != .None {
			return false, 0
		}
		start := time.now()
		for _ in 0..<reps {
			if err := c2r_inverse(&plan, output, restore); err != .None {
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

resolve_r2c_backend_autotuned :: proc(n: int, allocator: runtime.Allocator) -> Backend {
	if cached, ok := auto_r2c_cache_lookup(n); ok {
		return cached
	}

	best := Backend.Cooley_Tukey
	best_elapsed := i64(-1)
	candidates := [2]Backend{.Cooley_Tukey, .Split_Radix}
	for cand in candidates {
		ok, elapsed := autotune_measure_r2c_backend_inverse(n, cand, allocator)
		if !ok {
			continue
		}
		if best_elapsed < 0 || elapsed < best_elapsed {
			best_elapsed = elapsed
			best = cand
		}
	}
	auto_r2c_cache_store(n, best)
	return best
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

	base_threads := resolve_thread_count(options.threads, n)
	profile := current_cpu_profile()
	if base_threads <= 1 {
		if cand, ok := resolve_small_c2c_single_candidate(n); ok {
			result.backend = cand.backend
			if options.cooley_radix == 0 {
				result.cooley_radix = cand.radix
			}
			if options.threads <= 0 {
				result.threads = cand.threads
			}
			return result
		}
	}
	should_autotune := false
	if base_threads <= 1 {
		switch profile.tier {
		case .AVX512_Class:
			// Small and mid sizes are more stable with the radix heuristic
			// than with short-sample autotune on recent SIMD CPUs.
			should_autotune = n >= 16384 && n <= AUTO_C2C_TUNE_SINGLE_MAX_N
		case .AVX2_FMA_Class, .AVX2_Class:
			should_autotune = n >= 16384 && n <= AUTO_C2C_TUNE_SINGLE_MAX_N
		case .SSE_Class, .Scalar, .NEON_Class:
			should_autotune = n >= 2048 && n <= (1 << 18)
		}
	} else {
		if profile.supports_avx2 {
			should_autotune = n >= 1024 && n <= AUTO_C2C_TUNE_MULTI_MAX_N
		} else {
			should_autotune = n >= 4096 && n <= (1 << 12)
		}
	}
	if !should_autotune {
		return result
	}

	cache_key := Auto_C2C_Cache_Key{
		n = n,
		threads = base_threads,
		requested_cooley_radix = options.cooley_radix,
		store_inverse_twiddles = options.store_inverse_twiddles,
		store_bitrev_table = options.store_bitrev_table,
	}
	if cached_best, ok := auto_c2c_cache_lookup(cache_key); ok {
		result.backend = cached_best.backend
		if options.cooley_radix == 0 {
			result.cooley_radix = cached_best.radix
		}
		if options.threads <= 0 {
			result.threads = cached_best.threads
		}
		return result
	}

	log2_n := log2_exact(n)
	base_backend := resolve_backend_for_size_and_threads(n, .Auto, base_threads, allocator)
	base_radix := resolve_cooley_radix(n, log2_n, options.cooley_radix, base_threads)
	best := auto_candidate_canonicalize(n, Auto_C2C_Candidate{
		backend = base_backend,
		radix = base_radix,
		threads = base_threads,
	})

	candidates: [8]Auto_C2C_Candidate
	candidate_count := 0
	auto_candidate_push_unique(n, &candidates, &candidate_count, best)
	if base_threads <= 1 {
		auto_candidate_push_unique(n, &candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 2, threads = 1})
		if (log2_n & 1) == 0 {
			auto_candidate_push_unique(n, &candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 4, threads = 1})
		}
		if n >= 512 {
			auto_candidate_push_unique(n, &candidates, &candidate_count, Auto_C2C_Candidate{backend = .Split_Radix, radix = 2, threads = 1})
		}
	} else {
		if (log2_n & 1) == 0 {
			auto_candidate_push_unique(n, &candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 4, threads = base_threads})
		}
		if base_threads > 1 {
			auto_candidate_push_unique(n, &candidates, &candidate_count, Auto_C2C_Candidate{backend = .Cooley_Tukey, radix = 2, threads = base_threads})
		}
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
	auto_c2c_cache_store(cache_key, best)

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

r2c_plan_estimate_bytes :: proc(n: int, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	if n < 2 {
		return 0
	}
	if is_power_of_two(n) && (n & 1) == 0 {
		half_n := n / 2
		bytes := c2c_plan_estimate_bytes(half_n, backend, store_inverse_twiddles, store_bitrev_table)
		bytes += (half_n/2 + 1) * size_of(complex128)
		bytes += (half_n/2 + 1) * size_of(complex128)
		bytes += half_n * size_of(complex128)
		return bytes
	}
	return c2c_plan_estimate_bytes(n, backend, store_inverse_twiddles, store_bitrev_table) + n*size_of(complex128)
}

r2r_plan_estimate_bytes :: proc(n: int, kind: R2R_Kind, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	_ = kind
	if n < 1 {
		return 0
	}
	bytes := n * size_of(complex128)
	if is_power_of_two(n) {
		bytes += r2c_plan_estimate_bytes(n, backend, store_inverse_twiddles, store_bitrev_table)
		bytes += n * size_of(f64)
		bytes += r2c_output_len(n) * size_of(complex128)
		return bytes
	}
	bytes += c2c_plan_estimate_bytes(n, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += n * size_of(complex128)
	return bytes
}

c2c_plan_heap_bytes :: proc(plan: ^C2C_Plan) -> int {
	if plan == nil || !plan.initialized {
		return 0
	}

	bytes := 0
	bytes += len(plan.bitrev) * size_of(u32)
	bytes += len(plan.digitrev4) * size_of(u32)
	bytes += len(plan.twiddles) * size_of(complex128)
	bytes += len(plan.twiddles_inv) * size_of(complex128)
	bytes += len(plan.scratch) * size_of(complex128)
	bytes += len(plan.parallel_threads) * size_of(rawptr)
	bytes += len(plan.parallel_ctx) * size_of(C2C_Parallel_Worker_Context)
	bytes += len(plan.bluestein_m) * size_of(complex128)
	bytes += len(plan.bluestein_chirp) * size_of(complex128)
	bytes += len(plan.bluestein_b_fft) * size_of(complex128)
	bytes += len(plan.radix2_simd_stages) * size_of(Radix2_SIMD_Stage)
	bytes += len(plan.radix4_simd_stages) * size_of(Radix4_SIMD_Stage)

	for stage in plan.radix2_simd_stages {
		bytes += len(stage.tw) * size_of(Radix2_Twiddle_Pair_Pack)
		bytes += len(stage.tw_inv) * size_of(Radix2_Twiddle_Pair_Pack)
	}
	for stage in plan.radix4_simd_stages {
		bytes += len(stage.tw) * size_of(Radix4_Twiddle_Triple_Pack)
	}
	if plan.bluestein_conv_plan != nil {
		bytes += size_of(C2C_Plan)
		bytes += c2c_plan_heap_bytes(plan.bluestein_conv_plan)
	}

	return bytes
}

r2c_plan_heap_bytes :: proc(plan: ^R2C_Plan) -> int {
	if plan == nil || !plan.initialized {
		return 0
	}
	bytes := 0
	bytes += c2c_plan_heap_bytes(&plan.c2c)
	bytes += len(plan.twiddles) * size_of(complex128)
	bytes += len(plan.twiddles_inv) * size_of(complex128)
	bytes += len(plan.scratch) * size_of(complex128)
	return bytes
}

r2r_plan_heap_bytes :: proc(plan: ^R2R_Plan) -> int {
	if plan == nil || !plan.initialized {
		return 0
	}
	bytes := 0
	bytes += r2c_plan_heap_bytes(&plan.r2c)
	bytes += c2c_plan_heap_bytes(&plan.c2c)
	bytes += len(plan.twiddles) * size_of(complex128)
	bytes += len(plan.scratch) * size_of(f64)
	bytes += len(plan.freq_scratch) * size_of(complex128)
	bytes += len(plan.complex_scratch) * size_of(complex128)
	return bytes
}

c2c_plan_2d_estimate_bytes :: proc(rows, cols: int, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	if rows < 1 || cols < 1 {
		return 0
	}
	bytes := 0
	bytes += c2c_plan_estimate_bytes(cols, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += c2c_plan_estimate_bytes(rows, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += rows * c2c_2d_recommended_scratch_cols(rows, cols) * size_of(complex128)
	return bytes
}

r2c_plan_2d_estimate_bytes :: proc(rows, cols: int, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	if rows < 1 || cols < 2 {
		return 0
	}
	freq_cols := r2c_2d_output_cols(cols)
	bytes := 0
	bytes += r2c_plan_estimate_bytes(cols, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += c2c_plan_estimate_bytes(rows, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += rows * c2c_2d_recommended_scratch_cols(rows, freq_cols) * size_of(complex128)
	bytes += rows * freq_cols * size_of(complex128)
	return bytes
}

r2r_plan_2d_estimate_bytes :: proc(rows, cols: int, kind: R2R_Kind, backend: Backend, store_inverse_twiddles := true, store_bitrev_table := true) -> int {
	if rows < 1 || cols < 1 {
		return 0
	}
	bytes := 0
	bytes += r2r_plan_estimate_bytes(cols, kind, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += r2r_plan_estimate_bytes(rows, kind, backend, store_inverse_twiddles, store_bitrev_table)
	bytes += rows * r2r_2d_recommended_scratch_cols(rows, cols) * size_of(f64)
	return bytes
}

c2c_plan_2d_heap_bytes :: proc(plan: ^C2C_Plan_2D) -> int {
	if plan == nil || !plan.initialized {
		return 0
	}
	bytes := 0
	bytes += c2c_plan_heap_bytes(&plan.row_plan)
	bytes += c2c_plan_heap_bytes(&plan.col_plan)
	bytes += len(plan.scratch) * size_of(complex128)
	bytes += len(plan.parallel_threads) * size_of(rawptr)
	bytes += len(plan.parallel_ctx) * size_of(C2C_2D_Parallel_Worker_Context)
	bytes += len(plan.parallel_scratch) * size_of(complex128)
	return bytes
}

r2c_plan_2d_heap_bytes :: proc(plan: ^R2C_Plan_2D) -> int {
	if plan == nil || !plan.initialized {
		return 0
	}
	bytes := 0
	bytes += r2c_plan_heap_bytes(&plan.row_plan)
	bytes += c2c_plan_heap_bytes(&plan.col_plan)
	bytes += len(plan.scratch) * size_of(complex128)
	bytes += len(plan.freq_scratch) * size_of(complex128)
	return bytes
}

r2r_plan_2d_heap_bytes :: proc(plan: ^R2R_Plan_2D) -> int {
	if plan == nil || !plan.initialized {
		return 0
	}
	bytes := 0
	bytes += r2r_plan_heap_bytes(&plan.row_plan)
	bytes += r2r_plan_heap_bytes(&plan.col_plan)
	bytes += len(plan.scratch) * size_of(f64)
	bytes += len(plan.parallel_threads) * size_of(rawptr)
	bytes += len(plan.parallel_ctx) * size_of(R2R_2D_Parallel_Worker_Context)
	bytes += len(plan.parallel_block_scratch) * size_of(f64)
	bytes += len(plan.parallel_real_scratch) * size_of(f64)
	bytes += len(plan.parallel_freq_scratch) * size_of(complex128)
	bytes += len(plan.parallel_complex_scratch) * size_of(complex128)
	return bytes
}

resolve_backend_for_size :: proc(n: int, requested: Backend, allocator: runtime.Allocator) -> Backend {
	_ = allocator
	if requested != .Auto {
		return requested
	}
	profile := current_cpu_profile()
	if profile.supports_avx2 {
		log2_n := log2_exact(n)
		// On recent SIMD CPUs, split-radix can beat Cooley-Tukey at a few
		// small odd-shuffle-heavy sizes because it avoids some permutation
		// and twiddle setup overhead.
			if (n >= 4 && n <= 32) || n == 128 || n == 512 {
			return .Split_Radix
		}
		// Mid and large power-of-two sizes are not monotonic here:
		// odd powers in the mid-large band tend to favor split-radix,
		// while larger sizes swing back toward Cooley-Tukey.
		if n >= (1 << 13) && n <= (1 << 17) {
			if (log2_n & 1) != 0 {
				return .Split_Radix
			}
			return .Cooley_Tukey
		}
		if n >= (1 << 18) {
			return .Cooley_Tukey
		}
	}
	if !profile.supports_avx2 {
		if n >= (1 << 13) {
			return .Split_Radix
		}
		return .Cooley_Tukey
	}
	if profile.supports_avx512 {
		if n >= (1 << 16) {
			return .Split_Radix
		}
		return .Cooley_Tukey
	}
	if profile.supports_fma {
		if n >= (1 << 16) {
			return .Split_Radix
		}
		return .Cooley_Tukey
	}
	if n >= (1 << 16) {
		return .Split_Radix
	}
	return .Cooley_Tukey
}

resolve_backend_for_size_and_threads :: proc(n: int, requested: Backend, num_threads: int, allocator: runtime.Allocator) -> Backend {
	if requested != .Auto {
		return requested
	}
	if num_threads > 1 {
		return .Cooley_Tukey
	}
	return resolve_backend_for_size(n, requested, allocator)
}

resolve_r2c_backend_for_size :: #force_inline proc(n: int, requested: Backend, allocator: runtime.Allocator) -> Backend {
	if requested != .Auto {
		return requested
	}
	if is_power_of_two(n) {
		// R2C/C2R shares one backend across forward and inverse.
		// The fastest backend is size-sensitive here:
		// 2^17 still favors Cooley-Tukey, and 2^18+ swings back to split-radix.
		if n >= (1 << 18) {
			return .Split_Radix
		}
		return .Cooley_Tukey
	}
	return resolve_backend_for_size_and_threads(max(n/2, 1), .Auto, 1, allocator)
}

resolve_r2r_backend_for_size :: #force_inline proc(n: int, kind: R2R_Kind, requested: Backend, allocator: runtime.Allocator) -> Backend {
	if requested != .Auto {
		return requested
	}
	_ = kind
	return resolve_backend_for_size_and_threads(n, .Auto, 1, allocator)
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
	if plan.twiddles_inv != nil {
		delete(plan.twiddles_inv, plan.allocator)
	}
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	plan^ = {}
}

r2r_plan_destroy :: proc(plan: ^R2R_Plan) {
	r2c_plan_destroy(&plan.r2c)
	c2c_plan_destroy(&plan.c2c)
	if plan.twiddles != nil {
		delete(plan.twiddles, plan.allocator)
	}
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	if plan.freq_scratch != nil {
		delete(plan.freq_scratch, plan.allocator)
	}
	if plan.complex_scratch != nil {
		delete(plan.complex_scratch, plan.allocator)
	}
	plan^ = {}
}

r2c_plan_2d_destroy :: proc(plan: ^R2C_Plan_2D) {
	r2c_plan_destroy(&plan.row_plan)
	c2c_plan_destroy(&plan.col_plan)
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	if plan.freq_scratch != nil {
		delete(plan.freq_scratch, plan.allocator)
	}
	plan^ = {}
}

r2r_plan_2d_destroy :: proc(plan: ^R2R_Plan_2D) {
	r2r_2d_parallel_shutdown(plan)
	r2r_plan_destroy(&plan.row_plan)
	r2r_plan_destroy(&plan.col_plan)
	if plan.scratch != nil {
		delete(plan.scratch, plan.allocator)
	}
	plan^ = {}
}

c2c_2d_parallel_worker_proc :: proc(t: ^thread.Thread) {
	ctx := (^C2C_2D_Parallel_Worker_Context)(t.data)
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

		worker_count := plan.parallel_worker_count
		switch mode {
		case .Rows_Forward, .Rows_Inverse:
			chunk := (plan.rows + worker_count - 1) / worker_count
			row_start := worker_index * chunk
			row_end := min(row_start + chunk, plan.rows)
			for r in row_start..<row_end {
				row := plan.parallel_data[r*plan.cols:][:plan.cols]
				if mode == .Rows_Forward {
					_ = c2c_forward_in_place(&plan.row_plan, row)
				} else {
					_ = c2c_inverse_in_place(&plan.row_plan, row)
				}
			}
		case .Cols_Forward, .Cols_Inverse:
			scratch_cols := c2c_2d_scratch_cols(plan, plan.scratch)
			block_count := (plan.cols + scratch_cols - 1) / scratch_cols
			chunk := (block_count + worker_count - 1) / worker_count
			block_start := worker_index * chunk
			block_end := min(block_start + chunk, block_count)
			scratch_stride := plan.rows * scratch_cols
			scratch := plan.parallel_scratch[worker_index*scratch_stride:][:scratch_stride]
			for block_i in block_start..<block_end {
				c0 := block_i * scratch_cols
				block_cols := min(scratch_cols, plan.cols-c0)
				for r in 0..<plan.rows {
					row := plan.parallel_data[r*plan.cols+c0:][:block_cols]
					#no_bounds_check for bc in 0..<block_cols {
						scratch[bc*plan.rows+r] = row[bc]
					}
				}
				for bc in 0..<block_cols {
					col := scratch[bc*plan.rows:][:plan.rows]
					if mode == .Cols_Forward {
						_ = c2c_forward_in_place(&plan.col_plan, col)
					} else {
						_ = c2c_inverse_in_place(&plan.col_plan, col)
					}
				}
				for r in 0..<plan.rows {
					row := plan.parallel_data[r*plan.cols+c0:][:block_cols]
					#no_bounds_check for bc in 0..<block_cols {
						row[bc] = scratch[bc*plan.rows+r]
					}
				}
			}
		case .Idle, .Stop:
		}
		sync.barrier_wait(&plan.parallel_done)
	}
}

r2r_2d_parallel_worker_proc :: proc(t: ^thread.Thread) {
	ctx := (^R2R_2D_Parallel_Worker_Context)(t.data)
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

		worker_count := plan.parallel_worker_count
		block_cols := r2r_2d_scratch_cols(plan, plan.scratch)
		block_stride := plan.rows * block_cols
		complex_stride := max(plan.rows, plan.cols)
		block_scratch := plan.parallel_block_scratch[worker_index*block_stride:][:block_stride]
		complex_scratch := plan.parallel_complex_scratch[worker_index*complex_stride:][:complex_stride]

		switch mode {
		case .Rows_Forward, .Rows_Inverse:
			chunk := (plan.rows + worker_count - 1) / worker_count
			row_start := worker_index * chunk
			row_end := min(row_start + chunk, plan.rows)
				for r in row_start..<row_end {
					row := plan.parallel_data[r*plan.cols:][:plan.cols]
					if mode == .Rows_Forward {
						_ = r2r_forward_with_buffers(&plan.row_plan, row, row, complex_scratch)
					} else {
						_ = r2r_inverse_with_buffers(&plan.row_plan, row, row, complex_scratch)
					}
				}
		case .Cols_Forward, .Cols_Inverse:
			block_count := (plan.cols + block_cols - 1) / block_cols
			chunk := (block_count + worker_count - 1) / worker_count
			block_start := worker_index * chunk
			block_end := min(block_start + chunk, block_count)
			for block_i in block_start..<block_end {
				c0 := block_i * block_cols
				current_block_cols := min(block_cols, plan.cols-c0)
				for r in 0..<plan.rows {
					row := plan.parallel_data[r*plan.cols+c0:][:current_block_cols]
					#no_bounds_check for bc in 0..<current_block_cols {
						block_scratch[bc*plan.rows+r] = row[bc]
					}
				}
					for bc in 0..<current_block_cols {
						col := block_scratch[bc*plan.rows:][:plan.rows]
						if mode == .Cols_Forward {
							_ = r2r_forward_with_buffers(&plan.col_plan, col, col, complex_scratch)
						} else {
							_ = r2r_inverse_with_buffers(&plan.col_plan, col, col, complex_scratch)
						}
					}
				for r in 0..<plan.rows {
					row := plan.parallel_data[r*plan.cols+c0:][:current_block_cols]
					#no_bounds_check for bc in 0..<current_block_cols {
						row[bc] = block_scratch[bc*plan.rows+r]
					}
				}
			}
		case .Idle, .Stop:
		}
		sync.barrier_wait(&plan.parallel_done)
	}
}

c2c_2d_parallel_should_enable :: #force_inline proc(plan: ^C2C_Plan_2D) -> bool {
	if plan.rows*plan.cols < C2C_2D_PARALLEL_MIN_POINTS {
		return false
	}
	if plan.row_plan.backend != .Cooley_Tukey || plan.col_plan.backend != .Cooley_Tukey {
		return false
	}
	if plan.row_plan.uses_bluestein || plan.col_plan.uses_bluestein {
		return false
	}
	if plan.row_plan.num_threads > 1 || plan.col_plan.num_threads > 1 {
		return false
	}
	if plan.row_plan.parallel_enabled || plan.col_plan.parallel_enabled {
		return false
	}
	return true
}

r2r_plan_parallel_safe :: #force_inline proc(plan: ^R2R_Plan) -> bool {
	if plan.c2c.backend != .Cooley_Tukey || plan.c2c.uses_bluestein {
		return false
	}
	if plan.c2c.num_threads > 1 || plan.c2c.parallel_enabled {
		return false
	}
	if !plan.uses_full_c2c {
		if plan.r2c.c2c.backend != .Cooley_Tukey || plan.r2c.c2c.uses_bluestein {
			return false
		}
		if plan.r2c.c2c.num_threads > 1 || plan.r2c.c2c.parallel_enabled {
			return false
		}
	}
	return true
}

r2r_2d_parallel_should_enable :: #force_inline proc(plan: ^R2R_Plan_2D) -> bool {
	if plan.rows*plan.cols < C2C_2D_PARALLEL_MIN_POINTS {
		return false
	}
	if !r2r_plan_parallel_safe(&plan.row_plan) || !r2r_plan_parallel_safe(&plan.col_plan) {
		return false
	}
	return true
}

c2c_2d_parallel_init :: proc(plan: ^C2C_Plan_2D) -> Error {
	if !c2c_2d_parallel_should_enable(plan) {
		return .None
	}

	scratch_cols := c2c_2d_scratch_cols(plan, plan.scratch)
	block_count := (plan.cols + scratch_cols - 1) / scratch_cols
	task_count := max(plan.rows, block_count)
	worker_count := min(resolve_thread_count(0, plan.rows*plan.cols), C2C_2D_PARALLEL_MAX_WORKERS)
	worker_count = min(worker_count, task_count)
	if worker_count <= 1 {
		return .None
	}

	threads, threads_err := make([]^thread.Thread, worker_count, plan.allocator)
	if threads_err != .None {
		return .Allocation_Failed
	}
	ctx, ctx_err := make([]C2C_2D_Parallel_Worker_Context, worker_count, plan.allocator)
	if ctx_err != .None {
		delete(threads, plan.allocator)
		return .Allocation_Failed
	}

	scratch_stride := plan.rows * scratch_cols
	parallel_scratch, ps_err := make([]complex128, worker_count*scratch_stride, plan.allocator)
	if ps_err != .None {
		delete(ctx, plan.allocator)
		delete(threads, plan.allocator)
		return .Allocation_Failed
	}

	created := 0
	for i in 0..<worker_count {
		ctx[i] = C2C_2D_Parallel_Worker_Context{plan = plan, worker_index = i}
		t := thread.create(c2c_2d_parallel_worker_proc)
		if t == nil {
			for j in 0..<created {
				thread.destroy(threads[j])
			}
			delete(parallel_scratch, plan.allocator)
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
	plan.parallel_scratch = parallel_scratch
	sync.barrier_init(&plan.parallel_start, worker_count + 1)
	sync.barrier_init(&plan.parallel_done, worker_count + 1)
	plan.parallel_enabled = true

	for t in plan.parallel_threads {
		thread.start(t)
	}
	return .None
}

r2r_2d_parallel_init :: proc(plan: ^R2R_Plan_2D) -> Error {
	if !r2r_2d_parallel_should_enable(plan) {
		return .None
	}

	scratch_cols := r2r_2d_scratch_cols(plan, plan.scratch)
	block_count := (plan.cols + scratch_cols - 1) / scratch_cols
	task_count := max(plan.rows, block_count)
	worker_count := min(resolve_thread_count(0, plan.rows*plan.cols), C2C_2D_PARALLEL_MAX_WORKERS)
	worker_count = min(worker_count, task_count)
	if worker_count <= 1 {
		return .None
	}

	threads, threads_err := make([]^thread.Thread, worker_count, plan.allocator)
	if threads_err != .None {
		return .Allocation_Failed
	}
	ctx, ctx_err := make([]R2R_2D_Parallel_Worker_Context, worker_count, plan.allocator)
	if ctx_err != .None {
		delete(threads, plan.allocator)
		return .Allocation_Failed
	}

	block_stride := plan.rows * scratch_cols
	complex_stride := max(plan.rows, plan.cols)

	block_scratch, bs_err := make([]f64, worker_count*block_stride, plan.allocator)
	if bs_err != .None {
		delete(ctx, plan.allocator)
		delete(threads, plan.allocator)
		return .Allocation_Failed
	}
	complex_scratch, cs_err := make([]complex128, worker_count*complex_stride, plan.allocator)
	if cs_err != .None {
		delete(block_scratch, plan.allocator)
		delete(ctx, plan.allocator)
		delete(threads, plan.allocator)
		return .Allocation_Failed
	}

	created := 0
	for i in 0..<worker_count {
		ctx[i] = R2R_2D_Parallel_Worker_Context{plan = plan, worker_index = i}
		t := thread.create(r2r_2d_parallel_worker_proc)
		if t == nil {
			for j in 0..<created {
				thread.destroy(threads[j])
			}
			delete(complex_scratch, plan.allocator)
			delete(block_scratch, plan.allocator)
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
	plan.parallel_block_scratch = block_scratch
	plan.parallel_complex_scratch = complex_scratch
	sync.barrier_init(&plan.parallel_start, worker_count + 1)
	sync.barrier_init(&plan.parallel_done, worker_count + 1)
	plan.parallel_enabled = true

	for t in plan.parallel_threads {
		thread.start(t)
	}
	return .None
}

c2c_2d_parallel_shutdown :: proc(plan: ^C2C_Plan_2D) {
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
		if plan.parallel_scratch != nil {
			delete(plan.parallel_scratch, plan.allocator)
		}
		plan.parallel_ctx = nil
		plan.parallel_threads = nil
		plan.parallel_scratch = nil
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
	delete(plan.parallel_scratch, plan.allocator)
	plan.parallel_threads = nil
	plan.parallel_ctx = nil
	plan.parallel_scratch = nil
	plan.parallel_worker_count = 0
	plan.parallel_enabled = false
	plan.parallel_mode = .Idle
}

r2r_2d_parallel_shutdown :: proc(plan: ^R2R_Plan_2D) {
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
		if plan.parallel_block_scratch != nil {
			delete(plan.parallel_block_scratch, plan.allocator)
		}
		if plan.parallel_real_scratch != nil {
			delete(plan.parallel_real_scratch, plan.allocator)
		}
		if plan.parallel_freq_scratch != nil {
			delete(plan.parallel_freq_scratch, plan.allocator)
		}
		if plan.parallel_complex_scratch != nil {
			delete(plan.parallel_complex_scratch, plan.allocator)
		}
		plan.parallel_ctx = nil
		plan.parallel_threads = nil
		plan.parallel_block_scratch = nil
		plan.parallel_real_scratch = nil
		plan.parallel_freq_scratch = nil
		plan.parallel_complex_scratch = nil
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
	delete(plan.parallel_block_scratch, plan.allocator)
	delete(plan.parallel_real_scratch, plan.allocator)
	delete(plan.parallel_freq_scratch, plan.allocator)
	delete(plan.parallel_complex_scratch, plan.allocator)
	plan.parallel_threads = nil
	plan.parallel_ctx = nil
	plan.parallel_block_scratch = nil
	plan.parallel_real_scratch = nil
	plan.parallel_freq_scratch = nil
	plan.parallel_complex_scratch = nil
	plan.parallel_worker_count = 0
	plan.parallel_enabled = false
	plan.parallel_mode = .Idle
}

c2c_plan_2d_destroy :: proc(plan: ^C2C_Plan_2D) {
	c2c_2d_parallel_shutdown(plan)
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
	resolved_threads := resolve_thread_count(effective_options.threads, n)
	resolved_backend := resolve_backend_for_size_and_threads(n, effective_options.backend, resolved_threads, allocator)

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
	plan.num_threads = resolved_threads
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
		if n >= (1 << 20) {
			if err := cooley_tukey_parallel_init(plan); err != .None {
				c2c_plan_destroy(plan)
				return err
			}
		}
	} else if resolved_backend == .Cooley_Tukey && (plan.cooley_radix == 8 || plan.cooley_radix == 16) {
		if err := cooley_tukey_radix4_simd_init(plan); err != .None {
			c2c_plan_destroy(plan)
			return err
		}
		if err := cooley_tukey_parallel_init(plan); err != .None {
			c2c_plan_destroy(plan)
			return err
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

	effective_backend := resolve_r2c_backend_for_size(n, backend, allocator)
	half_n := n / 2
	if is_power_of_two(n) {
		c2c_err := c2c_plan_init_with_backend(&plan.c2c, half_n, effective_backend, allocator)
		if c2c_err != .None {
			return c2c_err
		}

		twiddles, twiddle_err := make([]complex128, half_n/2+1, allocator)
		if twiddle_err != .None {
			r2c_plan_destroy(plan)
			return .Allocation_Failed
		}
		twiddles_inv, twiddle_inv_err := make([]complex128, half_n/2+1, allocator)
		if twiddle_inv_err != .None {
			delete(twiddles, allocator)
			r2c_plan_destroy(plan)
			return .Allocation_Failed
		}
		scratch, scratch_err := make([]complex128, half_n, allocator)
		if scratch_err != .None {
			delete(twiddles, allocator)
			delete(twiddles_inv, allocator)
			r2c_plan_destroy(plan)
			return .Allocation_Failed
		}

		#no_bounds_check for k in 0..<len(twiddles) {
			angle := -2.0 * math.PI * f64(k) / f64(n)
			s, c := math.sincos(angle)
			twiddles[k] = complex(c, s)
			twiddles_inv[k] = complex(c, -s)
		}

		plan.twiddles = twiddles
		plan.twiddles_inv = twiddles_inv
		plan.scratch = scratch
		plan.uses_full_c2c = false
	} else {
		c2c_err := c2c_plan_init_with_backend(&plan.c2c, n, effective_backend, allocator)
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

r2r_plan_init_with_backend :: proc(plan: ^R2R_Plan, n: int, kind: R2R_Kind, backend: Backend, allocator := context.allocator) -> Error {
	if n < 1 {
		return .Invalid_Length
	}
	r2r_plan_destroy(plan)

	effective_backend := resolve_r2r_backend_for_size(n, kind, backend, allocator)
	twiddles, twiddle_err := make([]complex128, n, allocator)
	if twiddle_err != .None {
		r2r_plan_destroy(plan)
		return .Allocation_Failed
	}
	if is_power_of_two(n) {
		r2c_err := r2c_plan_init_with_backend(&plan.r2c, n, effective_backend, allocator)
		if r2c_err != .None {
			delete(twiddles, allocator)
			r2r_plan_destroy(plan)
			return r2c_err
		}
		real_scratch, rs_err := make([]f64, n, allocator)
		if rs_err != .None {
			delete(twiddles, allocator)
			r2r_plan_destroy(plan)
			return .Allocation_Failed
		}
		freq_scratch, fs_err := make([]complex128, r2c_output_len(n), allocator)
		if fs_err != .None {
			delete(twiddles, allocator)
			delete(real_scratch, allocator)
			r2r_plan_destroy(plan)
			return .Allocation_Failed
		}
		plan.scratch = real_scratch
		plan.freq_scratch = freq_scratch
		plan.uses_full_c2c = false
	} else {
		c2c_err := c2c_plan_init_with_backend(&plan.c2c, n, effective_backend, allocator)
		if c2c_err != .None {
			delete(twiddles, allocator)
			return c2c_err
		}
		complex_scratch, cs_err := make([]complex128, n, allocator)
		if cs_err != .None {
			delete(twiddles, allocator)
			r2r_plan_destroy(plan)
			return .Allocation_Failed
		}
		plan.complex_scratch = complex_scratch
		plan.uses_full_c2c = true
	}

	switch kind {
	case .DCT_II:
		#no_bounds_check for k in 0..<n {
			angle := -math.PI * f64(k) / (2.0 * f64(n))
			s, c := math.sincos(angle)
			twiddles[k] = complex(c, s)
		}
	}

	plan.n = n
	plan.kind = kind
	plan.twiddles = twiddles
	plan.allocator = allocator
	plan.initialized = true
	return .None
}

r2r_plan_init :: proc(plan: ^R2R_Plan, n: int, kind: R2R_Kind, allocator := context.allocator) -> Error {
	return r2r_plan_init_with_backend(plan, n, kind, .Auto, allocator)
}

r2r_2d_effective_backend :: #force_inline proc(rows, cols: int, backend: Backend) -> Backend {
	if backend != .Auto {
		return backend
	}
	if rows*cols >= C2C_2D_PARALLEL_MIN_POINTS && is_power_of_two(rows) && is_power_of_two(cols) {
		return .Cooley_Tukey
	}
	return .Auto
}

r2r_plan_2d_init_with_backend :: proc(plan: ^R2R_Plan_2D, rows, cols: int, kind: R2R_Kind, backend: Backend, allocator := context.allocator) -> Error {
	if rows < 1 || cols < 1 {
		return .Invalid_Length
	}
	r2r_plan_2d_destroy(plan)

	effective_backend := r2r_2d_effective_backend(rows, cols, backend)

	row_err := r2r_plan_init_with_backend(&plan.row_plan, cols, kind, effective_backend, allocator)
	if row_err != .None {
		return row_err
	}
	col_err := r2r_plan_init_with_backend(&plan.col_plan, rows, kind, effective_backend, allocator)
	if col_err != .None {
		r2r_plan_destroy(&plan.row_plan)
		return col_err
	}

	scratch_cols := r2r_2d_recommended_scratch_cols(rows, cols)
	scratch, sc_err := make([]f64, rows*scratch_cols, allocator)
	if sc_err != .None {
		r2r_plan_2d_destroy(plan)
		return .Allocation_Failed
	}

	plan.rows = rows
	plan.cols = cols
	plan.kind = kind
	if plan.row_plan.c2c.backend == plan.col_plan.c2c.backend {
	plan.backend = plan.col_plan.c2c.backend
	} else {
		plan.backend = .Auto
	}
	plan.scratch = scratch
	plan.allocator = allocator
	plan.initialized = true
	if err := r2r_2d_parallel_init(plan); err != .None {
		r2r_plan_2d_destroy(plan)
		return err
	}
	return .None
}

r2r_plan_2d_init :: proc(plan: ^R2R_Plan_2D, rows, cols: int, kind: R2R_Kind, allocator := context.allocator) -> Error {
	return r2r_plan_2d_init_with_backend(plan, rows, cols, kind, .Auto, allocator)
}

r2c_plan_2d_init_with_backend :: proc(plan: ^R2C_Plan_2D, rows, cols: int, backend: Backend, allocator := context.allocator) -> Error {
	if rows < 1 || cols < 2 {
		return .Invalid_Length
	}
	r2c_plan_2d_destroy(plan)

	row_err := r2c_plan_init_with_backend(&plan.row_plan, cols, backend, allocator)
	if row_err != .None {
		return row_err
	}
	col_err := c2c_plan_init_with_backend(&plan.col_plan, rows, backend, allocator)
	if col_err != .None {
		r2c_plan_destroy(&plan.row_plan)
		return col_err
	}

	freq_cols := r2c_2d_output_cols(cols)
	scratch_cols := c2c_2d_recommended_scratch_cols(rows, freq_cols)
	scratch, sc_err := make([]complex128, rows*scratch_cols, allocator)
	if sc_err != .None {
		r2c_plan_2d_destroy(plan)
		return .Allocation_Failed
	}
	freq_scratch, fs_err := make([]complex128, rows*freq_cols, allocator)
	if fs_err != .None {
		delete(scratch, allocator)
		r2c_plan_2d_destroy(plan)
		return .Allocation_Failed
	}

	plan.rows = rows
	plan.cols = cols
	plan.freq_cols = freq_cols
	if plan.row_plan.c2c.backend == plan.col_plan.backend {
		plan.backend = plan.col_plan.backend
	} else {
		plan.backend = .Auto
	}
	plan.scratch = scratch
	plan.freq_scratch = freq_scratch
	plan.allocator = allocator
	plan.initialized = true
	return .None
}

r2c_plan_2d_init :: proc(plan: ^R2C_Plan_2D, rows, cols: int, allocator := context.allocator) -> Error {
	return r2c_plan_2d_init_with_backend(plan, rows, cols, .Auto, allocator)
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

	scratch_cols := c2c_2d_recommended_scratch_cols(rows, cols)
	scratch, sc_err := make([]complex128, rows*scratch_cols, allocator)
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
	if err := c2c_2d_parallel_init(plan); err != .None {
		c2c_plan_2d_destroy(plan)
		return err
	}
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
		if plan.cooley_radix == 8 {
			return cooley_tukey_forward_radix8_in_place(plan, data)
		}
		if plan.cooley_radix == 16 {
			return cooley_tukey_forward_radix16_in_place(plan, data)
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
		if plan.cooley_radix == 8 {
			return cooley_tukey_inverse_radix8_in_place(plan, data)
		}
		if plan.cooley_radix == 16 {
			return cooley_tukey_inverse_radix16_in_place(plan, data)
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

	// Use output as work buffer
	work := output[:plan.half_n]

	copy(([^]f64)(raw_data(work))[:2*plan.half_n], input)

	if err := c2c_forward_in_place(&plan.c2c, work); err != .None {
		return err
	}

	a0 := work[0]
	output[0] = complex(real(a0)+imag(a0), 0.0)
	output[plan.half_n] = complex(real(a0)-imag(a0), 0.0)

	quarter := plan.half_n / 2
	k := 1
	when FFT_USE_SIMD_KERNELS {
		v05 := simd.f64x4{0.5, 0.5, 0.5, 0.5}
		for ; k + 1 < quarter; k += 2 {
			mirror := plan.half_n - k
			// work[k], work[k+1]
			vk := intrinsics.unaligned_load(cast(^simd.f64x4)(&work[k]))
			// work[mirror], work[mirror-1] -> reverse to get work[mirror-1], work[mirror]
			vm_raw := intrinsics.unaligned_load(cast(^simd.f64x4)(&work[mirror-1]))
			vm := simd.shuffle(vm_raw, vm_raw, 2, 3, 0, 1)

			w := intrinsics.unaligned_load(cast(^simd.f64x4)(&plan.twiddles[k]))

			sum := vk + vm
			diff := vk - vm

			sum_r := simd.shuffle(sum, sum, 0, 0, 2, 2)
			sum_i := simd.shuffle(sum, sum, 1, 1, 3, 3)
			diff_r := simd.shuffle(diff, diff, 0, 0, 2, 2)
			diff_i := simd.shuffle(diff, diff, 1, 1, 3, 3)

			wr := simd.shuffle(w, w, 0, 0, 2, 2)
			wi := simd.shuffle(w, w, 1, 1, 3, 3)

			re_term := v05 * (wr*sum_i + wi*diff_r)
			im_base := v05 * (wi*sum_i - wr*diff_r)
			im_delta := v05 * diff_i

			res_k := simd.shuffle(v05*sum_r + re_term, im_base + im_delta, 0, 4, 2, 6)
			res_m := simd.shuffle(v05*sum_r - re_term, im_base - im_delta, 0, 4, 2, 6)

			intrinsics.unaligned_store(cast(^simd.f64x4)(&output[k]), res_k)
			// Store output[mirror-1], output[mirror] after reversing back
			res_m_rev := simd.shuffle(res_m, res_m, 2, 3, 0, 1)
			intrinsics.unaligned_store(cast(^simd.f64x4)(&output[mirror-1]), res_m_rev)
		}
	}

	for ; k < quarter; k += 1 {
		mirror := plan.half_n - k
		a := work[k]
		w := plan.twiddles[k]
		ar, ai := real(a), imag(a)
		b := work[mirror]
		br, bi := real(b), imag(b)
		wr, wi := real(w), imag(w)
		sum_r := ar + br
		sum_i := ai + bi
		diff_r := ar - br
		diff_i := ai - bi
		re_term := 0.5 * (wr*sum_i + wi*diff_r)
		im_base := 0.5 * (-wr*diff_r + wi*sum_i)
		im_delta := 0.5 * diff_i

		output[k] = complex(0.5*sum_r+re_term, im_base+im_delta)
		output[mirror] = complex(0.5*sum_r-re_term, im_base-im_delta)
	}
	if plan.half_n > 1 {
		k := quarter
		a := work[k]
		w := plan.twiddles[k]
		ar, ai := real(a), imag(a)
		wr, wi := real(w), imag(w)
		output[k] = complex(ar+wr*ai, wi*ai)
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

	// Use output as work buffer
	work := ([^]complex128)(raw_data(output))[:plan.half_n]

	x0 := input[0]
	xn2 := input[plan.half_n]
	work[0] = complex(
		0.5*(real(x0)+real(xn2)),
		0.5*(real(x0)-real(xn2)),
	)

	quarter := plan.half_n / 2
	k := 1
	when FFT_USE_SIMD_KERNELS {
		v05 := simd.f64x4{0.5, 0.5, 0.5, 0.5}
		for ; k + 1 < quarter; k += 2 {
			mirror := plan.half_n - k
			// input[k], input[k+1]
			vk := intrinsics.unaligned_load(cast(^simd.f64x4)(&input[k]))
			// input[mirror], input[mirror-1] -> reverse to get input[mirror-1], input[mirror]
			vm_raw := intrinsics.unaligned_load(cast(^simd.f64x4)(&input[mirror-1]))
			vm := simd.shuffle(vm_raw, vm_raw, 2, 3, 0, 1)

			w := intrinsics.unaligned_load(cast(^simd.f64x4)(&plan.twiddles_inv[k]))

			sum := vk + vm
			diff := vk - vm

			sum_r := simd.shuffle(sum, sum, 0, 0, 2, 2)
			sum_i := simd.shuffle(sum, sum, 1, 1, 3, 3)
			diff_r := simd.shuffle(diff, diff, 0, 0, 2, 2)
			diff_i := simd.shuffle(diff, diff, 1, 1, 3, 3)

			wr := simd.shuffle(w, w, 0, 0, 2, 2)
			wi := simd.shuffle(w, w, 1, 1, 3, 3)

			re_term := v05 * (wi*diff_r + wr*sum_i)
			im_base := v05 * (wr*diff_r - wi*sum_i)
			im_delta := v05 * diff_i

			res_k := simd.shuffle(v05*sum_r - re_term, im_base + im_delta, 0, 4, 2, 6)
			res_m := simd.shuffle(v05*sum_r + re_term, im_base - im_delta, 0, 4, 2, 6)

			intrinsics.unaligned_store(cast(^simd.f64x4)(&work[k]), res_k)
			// Store work[mirror-1], work[mirror] after reversing back
			res_m_rev := simd.shuffle(res_m, res_m, 2, 3, 0, 1)
			intrinsics.unaligned_store(cast(^simd.f64x4)(&work[mirror-1]), res_m_rev)
		}
	}

	for ; k < quarter; k += 1 {
		mirror := plan.half_n - k
		c := input[k]
		w := plan.twiddles_inv[k]
		cr, ci := real(c), imag(c)
		m := input[mirror]
		mr, mi := real(m), imag(m)
		wr, wi := real(w), imag(w)
		sum_r := cr + mr
		sum_i := ci + mi
		diff_r := cr - mr
		diff_i := ci - mi
		re_term := 0.5 * (wi*diff_r + wr*sum_i)
		im_base := 0.5 * (wr*diff_r - wi*sum_i)
		im_delta := 0.5 * diff_i

		work[k] = complex(0.5*sum_r-re_term, im_base+im_delta)
		work[mirror] = complex(0.5*sum_r+re_term, im_base-im_delta)
	}
	if plan.half_n > 1 {
		k := quarter
		c := input[k]
		w := plan.twiddles_inv[k]
		cr, ci := real(c), imag(c)
		wr, wi := real(w), imag(w)
		work[k] = complex(cr-wr*ci, -wi*ci)
	}

	if err := c2c_inverse_in_place(&plan.c2c, work); err != .None {
		return err
	}

	return .None
}

r2c_forward :: proc(plan: ^R2C_Plan, input: []f64, output: []complex128) -> Error {
	return r2c_forward_with_scratch(plan, input, output, plan.scratch)
}

c2r_inverse :: proc(plan: ^R2C_Plan, input: []complex128, output: []f64) -> Error {
	return c2r_inverse_with_scratch(plan, input, output, plan.scratch)
}

dct2_reorder_input :: proc(dst: []complex128, input: []f64) {
	n := len(input)
	half := n / 2
	#no_bounds_check for i in 0..<half {
		dst[i] = complex(input[2*i], 0.0)
		dst[n-1-i] = complex(input[2*i+1], 0.0)
	}
	if (n & 1) == 1 {
		dst[half] = complex(input[n-1], 0.0)
	}
}

dct2_reorder_input_real :: proc(dst: []f64, input: []f64) {
	n := len(input)
	half := n / 2
	#no_bounds_check for i in 0..<half {
		dst[i] = input[2*i]
		dst[n-1-i] = input[2*i+1]
	}
	if (n & 1) == 1 {
		dst[half] = input[n-1]
	}
}

dct2_unreorder_output :: proc(input: []complex128, output: []f64) {
	n := len(output)
	half := n / 2
	#no_bounds_check for i in 0..<half {
		output[2*i] = real(input[i])
		output[2*i+1] = real(input[n-1-i])
	}
	if (n & 1) == 1 {
		output[n-1] = real(input[half])
	}
}

dct2_unreorder_output_real :: proc(input: []f64, output: []f64) {
	n := len(output)
	half := n / 2
	#no_bounds_check for i in 0..<half {
		output[2*i] = input[i]
		output[2*i+1] = input[n-1-i]
	}
	if (n & 1) == 1 {
		output[n-1] = input[half]
	}
}

dct2_forward_from_reordered_spectrum :: proc(spectrum, twiddles: []complex128, output: []f64) {
	#no_bounds_check for k in 0..<len(output) {
		output[k] = real(spectrum[k] * twiddles[k])
	}
}

dct2_forward_from_half_spectrum :: proc(spectrum, twiddles: []complex128, output: []f64) {
	n := len(output)
	half_n := n / 2
	#no_bounds_check for k in 0..=half_n {
		output[k] = real(spectrum[k] * twiddles[k])
	}
	#no_bounds_check for k in half_n+1..<n {
		mirror := n - k
		output[k] = real(conj(spectrum[mirror]) * twiddles[k])
	}
}

dct2_inverse_build_reordered_spectrum :: proc(dst: []complex128, input: []f64, twiddles: []complex128) {
	n := len(input)
	dst[0] = complex(input[0], 0.0)
	#no_bounds_check for k in 1..<n {
		dst[k] = complex(input[k], -input[n-k]) * conj(twiddles[k])
	}
}

dct2_inverse_build_half_spectrum :: proc(dst: []complex128, input: []f64, twiddles: []complex128) {
	n := len(input)
	half_n := n / 2
	dst[0] = complex(input[0], 0.0)
	#no_bounds_check for k in 1..=half_n {
		dst[k] = complex(input[k], -input[n-k]) * conj(twiddles[k])
	}
}

r2r_forward_with_buffers :: proc(plan: ^R2R_Plan, input, output: []f64, complex_scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.n {
		return .Size_Mismatch
	}

	switch plan.kind {
	case .DCT_II:
		if !plan.uses_full_c2c {
			if len(plan.scratch) < plan.n || len(plan.freq_scratch) < r2c_output_len(plan.n) {
				return .Size_Mismatch
			}
			work_real := plan.scratch[:plan.n]
			work_freq := plan.freq_scratch[:r2c_output_len(plan.n)]
			dct2_reorder_input_real(work_real, input)
			if err := r2c_forward(&plan.r2c, work_real, work_freq); err != .None {
				return err
			}
			dct2_forward_from_half_spectrum(work_freq, plan.twiddles, output)
			return .None
		}
		if len(complex_scratch) < plan.n {
			return .Size_Mismatch
		}
		work := complex_scratch[:plan.n]
		dct2_reorder_input(work, input)
		if err := c2c_forward_in_place(&plan.c2c, work); err != .None {
			return err
		}
		dct2_forward_from_reordered_spectrum(work, plan.twiddles, output)
	}
	return .None
}

r2r_forward :: proc(plan: ^R2R_Plan, input: []f64, output: []f64) -> Error {
	return r2r_forward_with_buffers(plan, input, output, plan.complex_scratch)
}

r2r_inverse_with_buffers :: proc(plan: ^R2R_Plan, input, output: []f64, complex_scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.n || len(output) != plan.n {
		return .Size_Mismatch
	}

	switch plan.kind {
	case .DCT_II:
		if !plan.uses_full_c2c {
			if len(plan.scratch) < plan.n || len(plan.freq_scratch) < r2c_output_len(plan.n) {
				return .Size_Mismatch
			}
			work_freq := plan.freq_scratch[:r2c_output_len(plan.n)]
			work_real := plan.scratch[:plan.n]
			dct2_inverse_build_half_spectrum(work_freq, input, plan.twiddles)
			if err := c2r_inverse(&plan.r2c, work_freq, work_real); err != .None {
				return err
			}
			dct2_unreorder_output_real(work_real, output)
			return .None
		}
		if len(complex_scratch) < plan.n {
			return .Size_Mismatch
		}
		work := complex_scratch[:plan.n]
		dct2_inverse_build_reordered_spectrum(work, input, plan.twiddles)
		if err := c2c_inverse_in_place(&plan.c2c, work); err != .None {
			return err
		}
		dct2_unreorder_output(work, output)
	}
	return .None
}

r2r_inverse :: proc(plan: ^R2R_Plan, input: []f64, output: []f64) -> Error {
	return r2r_inverse_with_buffers(plan, input, output, plan.complex_scratch)
}

r2r_forward_in_place :: proc(plan: ^R2R_Plan, data: []f64) -> Error {
	return r2r_forward(plan, data, data)
}

r2r_inverse_in_place :: proc(plan: ^R2R_Plan, data: []f64) -> Error {
	return r2r_inverse(plan, data, data)
}

c2c_2d_recommended_scratch_cols :: #force_inline proc(rows, cols: int) -> int {
	if rows < 1 || cols < 1 {
		return 1
	}
	if rows*cols < C2C_2D_BLOCK_MIN_POINTS {
		return 1
	}
	// Prefer 4 columns for SIMD transposition if possible
	if cols >= 4 && (rows & 3) == 0 {
		return 4
	}
	target_bytes := C2C_2D_TARGET_SCRATCH_BYTES
	max_scratch_cols := C2C_2D_MAX_SCRATCH_COLS
	if rows*cols >= (1 << 20) {
		target_bytes *= 2
		max_scratch_cols = C2C_2D_LARGE_MAX_SCRATCH_COLS
	}
	target_elems := target_bytes / size_of(complex128)
	if target_elems < rows*2 {
		return 1
	}
	scratch_cols := target_elems / rows
	if scratch_cols < 2 {
		return 1
	}
	return min(cols, min(scratch_cols, max_scratch_cols))
}

r2r_2d_recommended_scratch_cols :: #force_inline proc(rows, cols: int) -> int {
	if rows < 1 || cols < 1 {
		return 1
	}
	if rows*cols < C2C_2D_BLOCK_MIN_POINTS {
		return 1
	}
	target_elems := C2C_2D_TARGET_SCRATCH_BYTES / size_of(f64)
	if target_elems < rows*2 {
		return 1
	}
	scratch_cols := target_elems / rows
	if scratch_cols < 2 {
		return 1
	}
	max_scratch_cols := C2C_2D_MAX_SCRATCH_COLS
	if rows*cols >= (1 << 16) {
		max_scratch_cols = C2C_2D_LARGE_MAX_SCRATCH_COLS
	}
	return min(cols, min(scratch_cols, max_scratch_cols))
}

r2r_2d_validate_real_grid :: #force_inline proc(rows, cols: int, data: []f64, scratch: []f64) -> Error {
	if len(data) != rows*cols || len(scratch) < rows {
		return .Size_Mismatch
	}
	return .None
}

validate_real_grid_strided :: #force_inline proc(rows, cols, row_stride: int, data: []f64) -> Error {
	if row_stride < cols || len(data) < rows*row_stride {
		return .Size_Mismatch
	}
	return .None
}

validate_complex_grid_strided :: #force_inline proc(rows, cols, row_stride: int, data: []complex128) -> Error {
	if row_stride < cols || len(data) < rows*row_stride {
		return .Size_Mismatch
	}
	return .None
}

r2r_2d_apply_rows_with_plan :: proc(plan: ^R2R_Plan, rows, cols: int, data: []f64, inverse: bool) -> Error {
	if len(data) != rows*cols {
		return .Size_Mismatch
	}
	for r in 0..<rows {
		row := data[r*cols:][:cols]
		if inverse {
			if err := r2r_inverse_in_place(plan, row); err != .None {
				return err
			}
		} else {
			if err := r2r_forward_in_place(plan, row); err != .None {
				return err
			}
		}
	}
	return .None
}

r2r_2d_apply_rows_with_plan_strided :: proc(plan: ^R2R_Plan, rows, cols: int, input: []f64, input_stride: int, output: []f64, output_stride: int, inverse: bool) -> Error {
	if err := validate_real_grid_strided(rows, cols, input_stride, input); err != .None {
		return err
	}
	if err := validate_real_grid_strided(rows, cols, output_stride, output); err != .None {
		return err
	}
	for r in 0..<rows {
		row_in := input[r*input_stride:][:cols]
		row_out := output[r*output_stride:][:cols]
		copy(row_out, row_in)
		if inverse {
			if err := r2r_inverse_in_place(plan, row_out); err != .None {
				return err
			}
		} else {
			if err := r2r_forward_in_place(plan, row_out); err != .None {
				return err
			}
		}
	}
	return .None
}

r2r_2d_transform_columns_with_plan :: proc(plan: ^R2R_Plan, rows, cols: int, data: []f64, scratch: []f64, inverse: bool) -> Error {
	if err := r2r_2d_validate_real_grid(rows, cols, data, scratch); err != .None {
		return err
	}

	scratch_cols := len(scratch) / rows
	if scratch_cols < 1 {
		return .Size_Mismatch
	}
	scratch_cols = min(scratch_cols, cols)
	if scratch_cols <= 1 {
		for c in 0..<cols {
			#no_bounds_check for r in 0..<rows {
				scratch[r] = data[r*cols+c]
			}
			if inverse {
				if err := r2r_inverse_in_place(plan, scratch[:rows]); err != .None {
					return err
				}
			} else {
				if err := r2r_forward_in_place(plan, scratch[:rows]); err != .None {
					return err
				}
			}
			#no_bounds_check for r in 0..<rows {
				data[r*cols+c] = scratch[r]
			}
		}
		return .None
	}

	for c0 := 0; c0 < cols; c0 += scratch_cols {
		block_cols := min(scratch_cols, cols-c0)
		for r in 0..<rows {
			row := data[r*cols+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				scratch[bc*rows+r] = row[bc]
			}
		}
		for bc in 0..<block_cols {
			col := scratch[bc*rows:][:rows]
			if inverse {
				if err := r2r_inverse_in_place(plan, col); err != .None {
					return err
				}
			} else {
				if err := r2r_forward_in_place(plan, col); err != .None {
					return err
				}
			}
		}
		for r in 0..<rows {
			row := data[r*cols+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				row[bc] = scratch[bc*rows+r]
			}
		}
	}

	return .None
}

r2r_2d_transform_columns_with_plan_strided :: proc(plan: ^R2R_Plan, rows, cols: int, data: []f64, row_stride: int, scratch: []f64, inverse: bool) -> Error {
	if err := validate_real_grid_strided(rows, cols, row_stride, data); err != .None {
		return err
	}
	if len(scratch) < rows {
		return .Size_Mismatch
	}

	scratch_cols := len(scratch) / rows
	if scratch_cols < 1 {
		return .Size_Mismatch
	}
	scratch_cols = min(scratch_cols, cols)
	if scratch_cols <= 1 {
		for c in 0..<cols {
			#no_bounds_check for r in 0..<rows {
				scratch[r] = data[r*row_stride+c]
			}
			if inverse {
				if err := r2r_inverse_in_place(plan, scratch[:rows]); err != .None {
					return err
				}
			} else {
				if err := r2r_forward_in_place(plan, scratch[:rows]); err != .None {
					return err
				}
			}
			#no_bounds_check for r in 0..<rows {
				data[r*row_stride+c] = scratch[r]
			}
		}
		return .None
	}

	for c0 := 0; c0 < cols; c0 += scratch_cols {
		block_cols := min(scratch_cols, cols-c0)
		for r in 0..<rows {
			row := data[r*row_stride+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				scratch[bc*rows+r] = row[bc]
			}
		}
		for bc in 0..<block_cols {
			col := scratch[bc*rows:][:rows]
			if inverse {
				if err := r2r_inverse_in_place(plan, col); err != .None {
					return err
				}
			} else {
				if err := r2r_forward_in_place(plan, col); err != .None {
					return err
				}
			}
		}
		for r in 0..<rows {
			row := data[r*row_stride+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				row[bc] = scratch[bc*rows+r]
			}
		}
	}

	return .None
}

c2c_2d_validate_complex_grid :: #force_inline proc(rows, cols: int, data: []complex128, scratch: []complex128) -> Error {
	if len(data) != rows*cols || len(scratch) < rows {
		return .Size_Mismatch
	}
	return .None
}

r2c_2d_validate_real_freq_grid :: #force_inline proc(rows, cols, freq_cols: int, input: []f64, output: []complex128, scratch: []complex128) -> Error {
	if len(input) != rows*cols || len(output) != rows*freq_cols || len(scratch) < rows {
		return .Size_Mismatch
	}
	return .None
}

c2c_2d_apply_rows_with_plan_strided :: proc(c2c_plan: ^C2C_Plan, rows, cols: int, input: []complex128, input_stride: int, output: []complex128, output_stride: int, inverse: bool) -> Error {
	if err := validate_complex_grid_strided(rows, cols, input_stride, input); err != .None {
		return err
	}
	if err := validate_complex_grid_strided(rows, cols, output_stride, output); err != .None {
		return err
	}
	for r in 0..<rows {
		row_in := input[r*input_stride:][:cols]
		row_out := output[r*output_stride:][:cols]
		copy(row_out, row_in)
		if inverse {
			if err := c2c_inverse_in_place(c2c_plan, row_out); err != .None {
				return err
			}
		} else {
			if err := c2c_forward_in_place(c2c_plan, row_out); err != .None {
				return err
			}
		}
	}
	return .None
}

c2r_2d_validate_real_freq_grid :: #force_inline proc(rows, cols, freq_cols: int, input: []complex128, output: []f64, scratch: []complex128) -> Error {
	if len(input) != rows*freq_cols || len(output) != rows*cols || len(scratch) < rows {
		return .Size_Mismatch
	}
	return .None
}

fft_2d_validate_exec_descriptor :: #force_inline proc(desc: ^FFT_2D_Exec_Descriptor) -> Error {
	switch desc.row_kind {
	case .C2C_Forward, .C2C_Inverse:
		return c2c_2d_validate_complex_grid(desc.rows, desc.cols, desc.complex_data, desc.scratch)
	case .R2C_Forward:
		return r2c_2d_validate_real_freq_grid(desc.rows, desc.cols, desc.freq_cols, desc.real_input, desc.complex_data, desc.scratch)
	case .C2R_Inverse:
		if err := c2r_2d_validate_real_freq_grid(desc.rows, desc.cols, desc.freq_cols, desc.complex_data, desc.real_output, desc.scratch); err != .None {
			return err
		}
		if len(desc.freq_scratch) < desc.rows*desc.freq_cols {
			return .Size_Mismatch
		}
	}
	return .None
}

c2c_2d_apply_rows_with_plan :: proc(c2c_plan: ^C2C_Plan, rows, cols: int, data: []complex128, inverse: bool) -> Error {
	if len(data) != rows*cols {
		return .Size_Mismatch
	}
	for r in 0..<rows {
		row := data[r*cols:][:cols]
		if inverse {
			if err := c2c_inverse_in_place(c2c_plan, row); err != .None {
				return err
			}
		} else {
			if err := c2c_forward_in_place(c2c_plan, row); err != .None {
				return err
			}
		}
	}
	return .None
}

r2c_2d_apply_rows_forward :: proc(r2c_plan: ^R2C_Plan, rows, cols, freq_cols: int, input: []f64, output: []complex128) -> Error {
	if len(input) != rows*cols || len(output) != rows*freq_cols {
		return .Size_Mismatch
	}
	for r in 0..<rows {
		row_in := input[r*cols:][:cols]
		row_out := output[r*freq_cols:][:freq_cols]
		if err := r2c_forward(r2c_plan, row_in, row_out); err != .None {
			return err
		}
	}
	return .None
}

r2c_2d_apply_rows_forward_strided :: proc(r2c_plan: ^R2C_Plan, rows, cols, freq_cols: int, input: []f64, input_stride: int, output: []complex128, output_stride: int) -> Error {
	if err := validate_real_grid_strided(rows, cols, input_stride, input); err != .None {
		return err
	}
	if err := validate_complex_grid_strided(rows, freq_cols, output_stride, output); err != .None {
		return err
	}
	for r in 0..<rows {
		row_in := input[r*input_stride:][:cols]
		row_out := output[r*output_stride:][:freq_cols]
		if err := r2c_forward(r2c_plan, row_in, row_out); err != .None {
			return err
		}
	}
	return .None
}

c2r_2d_apply_rows_inverse :: proc(r2c_plan: ^R2C_Plan, rows, cols, freq_cols: int, input: []complex128, output: []f64) -> Error {
	if len(input) != rows*freq_cols || len(output) != rows*cols {
		return .Size_Mismatch
	}
	for r in 0..<rows {
		row_in := input[r*freq_cols:][:freq_cols]
		row_out := output[r*cols:][:cols]
		if err := c2r_inverse(r2c_plan, row_in, row_out); err != .None {
			return err
		}
	}
	return .None
}

c2r_2d_apply_rows_inverse_strided :: proc(r2c_plan: ^R2C_Plan, rows, cols, freq_cols: int, input: []complex128, output: []f64, output_stride: int) -> Error {
	if len(input) != rows*freq_cols {
		return .Size_Mismatch
	}
	if err := validate_real_grid_strided(rows, cols, output_stride, output); err != .None {
		return err
	}
	for r in 0..<rows {
		row_in := input[r*freq_cols:][:freq_cols]
		row_out := output[r*output_stride:][:cols]
		if err := c2r_inverse(r2c_plan, row_in, row_out); err != .None {
			return err
		}
	}
	return .None
}

fft_2d_apply_row_transform :: proc(desc: ^FFT_2D_Exec_Descriptor) -> Error {
	switch desc.row_kind {
	case .C2C_Forward:
		return c2c_2d_apply_rows_with_plan(desc.c2c_plan, desc.rows, desc.cols, desc.complex_data, false)
	case .C2C_Inverse:
		return c2c_2d_apply_rows_with_plan(desc.c2c_plan, desc.rows, desc.cols, desc.complex_data, true)
	case .R2C_Forward:
		return r2c_2d_apply_rows_forward(desc.r2c_plan, desc.rows, desc.cols, desc.freq_cols, desc.real_input, desc.complex_data)
	case .C2R_Inverse:
		return c2r_2d_apply_rows_inverse(desc.r2c_plan, desc.rows, desc.cols, desc.freq_cols, desc.complex_data, desc.real_output)
	}
	return .None
}

fft_2d_complex_cols :: #force_inline proc(desc: ^FFT_2D_Exec_Descriptor) -> int {
	switch desc.row_kind {
	case .R2C_Forward, .C2R_Inverse:
		return desc.freq_cols
	case .C2C_Forward, .C2C_Inverse:
		return desc.cols
	}
	return desc.cols
}

fft_2d_apply_column_transform :: proc(desc: ^FFT_2D_Exec_Descriptor, data: []complex128) -> Error {
	complex_cols := fft_2d_complex_cols(desc)
	switch desc.col_kind {
	case .C2C_Forward:
		return c2c_2d_transform_columns_with_plan(desc.col_plan, desc.rows, complex_cols, data, desc.scratch, false)
	case .C2C_Inverse:
		return c2c_2d_transform_columns_with_plan(desc.col_plan, desc.rows, complex_cols, data, desc.scratch, true)
	case .R2C_Forward, .C2R_Inverse:
	}
	return .None
}

c2c_2d_scratch_cols :: #force_inline proc(plan: ^C2C_Plan_2D, scratch: []complex128) -> int {
	if plan.rows <= 0 {
		return 0
	}
	scratch_cols := len(scratch) / plan.rows
	if scratch_cols < 1 {
		return 0
	}
	return min(scratch_cols, plan.cols)
}

r2r_2d_scratch_cols :: #force_inline proc(plan: ^R2R_Plan_2D, scratch: []f64) -> int {
	if plan.rows <= 0 {
		return 0
	}
	scratch_cols := len(scratch) / plan.rows
	if scratch_cols < 1 {
		return 0
	}
	return min(scratch_cols, plan.cols)
}

c2c_2d_transform_columns_with_plan :: proc(c2c_plan: ^C2C_Plan, rows, cols: int, data: []complex128, scratch: []complex128, inverse: bool) -> Error {
	if err := c2c_2d_validate_complex_grid(rows, cols, data, scratch); err != .None {
		return err
	}

	scratch_cols := len(scratch) / rows
	if scratch_cols < 1 {
		return .Size_Mismatch
	}
	scratch_cols = min(scratch_cols, cols)
	if scratch_cols <= 1 {
		for c in 0..<cols {
			#no_bounds_check for r in 0..<rows {
				scratch[r] = data[r*cols+c]
			}
			if inverse {
				if err := c2c_inverse_in_place(c2c_plan, scratch[:rows]); err != .None {
					return err
				}
			} else {
				if err := c2c_forward_in_place(c2c_plan, scratch[:rows]); err != .None {
					return err
				}
			}
			#no_bounds_check for r in 0..<rows {
				data[r*cols+c] = scratch[r]
			}
		}
		return .None
	}

	for c0 := 0; c0 < cols; c0 += scratch_cols {
		block_cols := min(scratch_cols, cols-c0)
		r := 0
		when FFT_USE_SIMD_KERNELS {
			if block_cols == 4 {
				for ; r + 3 < rows; r += 4 {
					r0 := &data[(r+0)*cols+c0]
					r1 := &data[(r+1)*cols+c0]
					r2 := &data[(r+2)*cols+c0]
					r3 := &data[(r+3)*cols+c0]
					
					c0_ptr := &scratch[0*rows+r]
					c1_ptr := &scratch[1*rows+r]
					c2_ptr := &scratch[2*rows+r]
					c3_ptr := &scratch[3*rows+r]
					
					transpose_4x4_complex_block(
						([^]complex128)(r0), ([^]complex128)(r1), ([^]complex128)(r2), ([^]complex128)(r3),
						([^]complex128)(c0_ptr), ([^]complex128)(c1_ptr), ([^]complex128)(c2_ptr), ([^]complex128)(c3_ptr),
					)
				}
			}
		}
		for ; r < rows; r += 1 {
			row := data[r*cols+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				scratch[bc*rows+r] = row[bc]
			}
		}
		for bc in 0..<block_cols {
			col := scratch[bc*rows:][:rows]
			if inverse {
				if err := c2c_inverse_in_place(c2c_plan, col); err != .None {
					return err
				}
			} else {
				if err := c2c_forward_in_place(c2c_plan, col); err != .None {
					return err
				}
			}
		}
		r = 0
		when FFT_USE_SIMD_KERNELS {
			if block_cols == 4 {
				for ; r + 3 < rows; r += 4 {
					c0_ptr := &scratch[0*rows+r]
					c1_ptr := &scratch[1*rows+r]
					c2_ptr := &scratch[2*rows+r]
					c3_ptr := &scratch[3*rows+r]

					r0 := &data[(r+0)*cols+c0]
					r1 := &data[(r+1)*cols+c0]
					r2 := &data[(r+2)*cols+c0]
					r3 := &data[(r+3)*cols+c0]
					
					transpose_4x4_complex_block(
						([^]complex128)(c0_ptr), ([^]complex128)(c1_ptr), ([^]complex128)(c2_ptr), ([^]complex128)(c3_ptr),
						([^]complex128)(r0), ([^]complex128)(r1), ([^]complex128)(r2), ([^]complex128)(r3),
					)
				}
			}
		}
		for ; r < rows; r += 1 {
			row := data[r*cols+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				row[bc] = scratch[bc*rows+r]
			}
		}
	}

	return .None
}

c2c_2d_transform_columns_with_plan_strided :: proc(c2c_plan: ^C2C_Plan, rows, cols: int, data: []complex128, row_stride: int, scratch: []complex128, inverse: bool) -> Error {
	if err := validate_complex_grid_strided(rows, cols, row_stride, data); err != .None {
		return err
	}
	if len(scratch) < rows {
		return .Size_Mismatch
	}

	scratch_cols := len(scratch) / rows
	if scratch_cols < 1 {
		return .Size_Mismatch
	}
	scratch_cols = min(scratch_cols, cols)
	if scratch_cols <= 1 {
		for c in 0..<cols {
			#no_bounds_check for r in 0..<rows {
				scratch[r] = data[r*row_stride+c]
			}
			if inverse {
				if err := c2c_inverse_in_place(c2c_plan, scratch[:rows]); err != .None {
					return err
				}
			} else {
				if err := c2c_forward_in_place(c2c_plan, scratch[:rows]); err != .None {
					return err
				}
			}
			#no_bounds_check for r in 0..<rows {
				data[r*row_stride+c] = scratch[r]
			}
		}
		return .None
	}

	for c0 := 0; c0 < cols; c0 += scratch_cols {
		block_cols := min(scratch_cols, cols-c0)
		for r in 0..<rows {
			row := data[r*row_stride+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				scratch[bc*rows+r] = row[bc]
			}
		}
		for bc in 0..<block_cols {
			col := scratch[bc*rows:][:rows]
			if inverse {
				if err := c2c_inverse_in_place(c2c_plan, col); err != .None {
					return err
				}
			} else {
				if err := c2c_forward_in_place(c2c_plan, col); err != .None {
					return err
				}
			}
		}
		for r in 0..<rows {
			row := data[r*row_stride+c0:][:block_cols]
			#no_bounds_check for bc in 0..<block_cols {
				row[bc] = scratch[bc*rows+r]
			}
		}
	}

	return .None
}

fft_2d_desc_c2c :: #force_inline proc(plan: ^C2C_Plan_2D, axis_kind: FFT_2D_Axis_Transform, data, scratch: []complex128) -> FFT_2D_Exec_Descriptor {
	return FFT_2D_Exec_Descriptor{
		row_kind = axis_kind,
		col_kind = axis_kind,
		pass_order = .Row_Then_Col,
		rows = plan.rows,
		cols = plan.cols,
		freq_cols = plan.cols,
		c2c_plan = &plan.row_plan,
		col_plan = &plan.col_plan,
		complex_data = data,
		scratch = scratch,
	}
}

fft_2d_desc_r2c_forward :: #force_inline proc(plan: ^R2C_Plan_2D, input: []f64, output, scratch: []complex128) -> FFT_2D_Exec_Descriptor {
	return FFT_2D_Exec_Descriptor{
		row_kind = .R2C_Forward,
		col_kind = .C2C_Forward,
		pass_order = .Row_Then_Col,
		rows = plan.rows,
		cols = plan.cols,
		freq_cols = plan.freq_cols,
		r2c_plan = &plan.row_plan,
		col_plan = &plan.col_plan,
		complex_data = output,
		scratch = scratch,
		real_input = input,
	}
}

fft_2d_desc_c2r_inverse :: #force_inline proc(plan: ^R2C_Plan_2D, input, scratch, freq_scratch: []complex128, output: []f64) -> FFT_2D_Exec_Descriptor {
	return FFT_2D_Exec_Descriptor{
		row_kind = .C2R_Inverse,
		col_kind = .C2C_Inverse,
		pass_order = .Col_Then_Row,
		rows = plan.rows,
		cols = plan.cols,
		freq_cols = plan.freq_cols,
		r2c_plan = &plan.row_plan,
		col_plan = &plan.col_plan,
		complex_data = input,
		scratch = scratch,
		freq_scratch = freq_scratch,
		real_output = output,
	}
}

fft_2d_execute :: proc(desc: ^FFT_2D_Exec_Descriptor) -> Error {
	if err := fft_2d_validate_exec_descriptor(desc); err != .None {
		return err
	}

	switch desc.pass_order {
	case .Row_Then_Col:
		if err := fft_2d_apply_row_transform(desc); err != .None {
			return err
		}
		return fft_2d_apply_column_transform(desc, desc.complex_data)
	case .Col_Then_Row:
		freq_work := desc.freq_scratch[:desc.rows*fft_2d_complex_cols(desc)]
		copy(freq_work, desc.complex_data)
		if err := fft_2d_apply_column_transform(desc, freq_work); err != .None {
			return err
		}
		desc_copy := desc^
		desc_copy.complex_data = freq_work
		return fft_2d_apply_row_transform(&desc_copy)
	}

	return .None
}

c2c_2d_forward_columns :: proc(plan: ^C2C_Plan_2D, data: []complex128, scratch: []complex128) -> Error {
	return c2c_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.cols, data, scratch, false)
}

c2c_2d_inverse_columns :: proc(plan: ^C2C_Plan_2D, data: []complex128, scratch: []complex128) -> Error {
	return c2c_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.cols, data, scratch, true)
}

c2c_2d_parallel_run :: proc(plan: ^C2C_Plan_2D, data: []complex128, mode: C2C_2D_Parallel_Mode) {
	plan.parallel_data = data
	plan.parallel_mode = mode
	sync.barrier_wait(&plan.parallel_start)
	sync.barrier_wait(&plan.parallel_done)
	plan.parallel_mode = .Idle
}

r2r_2d_parallel_run :: proc(plan: ^R2R_Plan_2D, data: []f64, mode: R2R_2D_Parallel_Mode) {
	plan.parallel_data = data
	plan.parallel_mode = mode
	sync.barrier_wait(&plan.parallel_start)
	sync.barrier_wait(&plan.parallel_done)
	plan.parallel_mode = .Idle
}

c2c_2d_forward_in_place_with_scratch :: proc(plan: ^C2C_Plan_2D, data: []complex128, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	desc := fft_2d_desc_c2c(plan, .C2C_Forward, data, scratch)
	return fft_2d_execute(&desc)
}

c2c_2d_inverse_in_place_with_scratch :: proc(plan: ^C2C_Plan_2D, data: []complex128, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	desc := fft_2d_desc_c2c(plan, .C2C_Inverse, data, scratch)
	return fft_2d_execute(&desc)
}

c2c_2d_forward_in_place :: proc(plan: ^C2C_Plan_2D, data: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	if plan.parallel_enabled {
		c2c_2d_parallel_run(plan, data, .Rows_Forward)
		c2c_2d_parallel_run(plan, data, .Cols_Forward)
		return .None
	}
	return c2c_2d_forward_in_place_with_scratch(plan, data, plan.scratch)
}

c2c_2d_inverse_in_place :: proc(plan: ^C2C_Plan_2D, data: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(data) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	if plan.parallel_enabled {
		c2c_2d_parallel_run(plan, data, .Rows_Inverse)
		c2c_2d_parallel_run(plan, data, .Cols_Inverse)
		return .None
	}
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

c2c_2d_forward_strided_with_scratch :: proc(plan: ^C2C_Plan_2D, input: []complex128, input_row_stride: int, output: []complex128, output_row_stride: int, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}
	if err := c2c_2d_apply_rows_with_plan_strided(&plan.row_plan, plan.rows, plan.cols, input, input_row_stride, output, output_row_stride, false); err != .None {
		return err
	}
	return c2c_2d_transform_columns_with_plan_strided(&plan.col_plan, plan.rows, plan.cols, output, output_row_stride, scratch, false)
}

c2c_2d_inverse_strided_with_scratch :: proc(plan: ^C2C_Plan_2D, input: []complex128, input_row_stride: int, output: []complex128, output_row_stride: int, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}
	if err := c2c_2d_apply_rows_with_plan_strided(&plan.row_plan, plan.rows, plan.cols, input, input_row_stride, output, output_row_stride, true); err != .None {
		return err
	}
	return c2c_2d_transform_columns_with_plan_strided(&plan.col_plan, plan.rows, plan.cols, output, output_row_stride, scratch, true)
}

c2c_2d_forward_strided :: proc(plan: ^C2C_Plan_2D, input: []complex128, input_row_stride: int, output: []complex128, output_row_stride: int) -> Error {
	return c2c_2d_forward_strided_with_scratch(plan, input, input_row_stride, output, output_row_stride, plan.scratch)
}

c2c_2d_inverse_strided :: proc(plan: ^C2C_Plan_2D, input: []complex128, input_row_stride: int, output: []complex128, output_row_stride: int) -> Error {
	return c2c_2d_inverse_strided_with_scratch(plan, input, input_row_stride, output, output_row_stride, plan.scratch)
}

r2c_2d_forward_with_scratch :: proc(plan: ^R2C_Plan_2D, input: []f64, output: []complex128, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	desc := fft_2d_desc_r2c_forward(plan, input, output, scratch)
	return fft_2d_execute(&desc)
}

c2r_2d_inverse_with_scratch :: proc(plan: ^R2C_Plan_2D, input: []complex128, output: []f64, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	desc := fft_2d_desc_c2r_inverse(plan, input, scratch, plan.freq_scratch, output)
	return fft_2d_execute(&desc)
}

r2c_2d_forward :: proc(plan: ^R2C_Plan_2D, input: []f64, output: []complex128) -> Error {
	return r2c_2d_forward_with_scratch(plan, input, output, plan.scratch)
}

c2r_2d_inverse :: proc(plan: ^R2C_Plan_2D, input: []complex128, output: []f64) -> Error {
	return c2r_2d_inverse_with_scratch(plan, input, output, plan.scratch)
}

r2c_2d_forward_strided_with_scratch :: proc(plan: ^R2C_Plan_2D, input: []f64, input_row_stride: int, output: []complex128, output_row_stride: int, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}
	if err := r2c_2d_apply_rows_forward_strided(&plan.row_plan, plan.rows, plan.cols, plan.freq_cols, input, input_row_stride, output, output_row_stride); err != .None {
		return err
	}
	return c2c_2d_transform_columns_with_plan_strided(&plan.col_plan, plan.rows, plan.freq_cols, output, output_row_stride, scratch, false)
}

c2r_2d_inverse_strided_with_scratch :: proc(plan: ^R2C_Plan_2D, input: []complex128, input_row_stride: int, output: []f64, output_row_stride: int, scratch: []complex128) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if err := validate_complex_grid_strided(plan.rows, plan.freq_cols, input_row_stride, input); err != .None {
		return err
	}
	if err := validate_real_grid_strided(plan.rows, plan.cols, output_row_stride, output); err != .None {
		return err
	}
	if len(scratch) < plan.rows || len(plan.freq_scratch) < plan.rows*plan.freq_cols {
		return .Size_Mismatch
	}

	freq_work := plan.freq_scratch[:plan.rows*plan.freq_cols]
	for r in 0..<plan.rows {
		row_in := input[r*input_row_stride:][:plan.freq_cols]
		row_work := freq_work[r*plan.freq_cols:][:plan.freq_cols]
		copy(row_work, row_in)
	}
	if err := c2c_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.freq_cols, freq_work, scratch, true); err != .None {
		return err
	}
	return c2r_2d_apply_rows_inverse_strided(&plan.row_plan, plan.rows, plan.cols, plan.freq_cols, freq_work, output, output_row_stride)
}

r2c_2d_forward_strided :: proc(plan: ^R2C_Plan_2D, input: []f64, input_row_stride: int, output: []complex128, output_row_stride: int) -> Error {
	return r2c_2d_forward_strided_with_scratch(plan, input, input_row_stride, output, output_row_stride, plan.scratch)
}

c2r_2d_inverse_strided :: proc(plan: ^R2C_Plan_2D, input: []complex128, input_row_stride: int, output: []f64, output_row_stride: int) -> Error {
	return c2r_2d_inverse_strided_with_scratch(plan, input, input_row_stride, output, output_row_stride, plan.scratch)
}

r2r_2d_forward_with_scratch :: proc(plan: ^R2R_Plan_2D, input: []f64, output: []f64, scratch: []f64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}

	copy(output, input)
	if err := r2r_2d_apply_rows_with_plan(&plan.row_plan, plan.rows, plan.cols, output, false); err != .None {
		return err
	}
	return r2r_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.cols, output, scratch, false)
}

r2r_2d_inverse_with_scratch :: proc(plan: ^R2R_Plan_2D, input: []f64, output: []f64, scratch: []f64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}

	copy(output, input)
	if err := r2r_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.cols, output, scratch, true); err != .None {
		return err
	}
	return r2r_2d_apply_rows_with_plan(&plan.row_plan, plan.rows, plan.cols, output, true)
}

r2r_2d_forward :: proc(plan: ^R2R_Plan_2D, input: []f64, output: []f64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	copy(output, input)
	if plan.parallel_enabled {
		r2r_2d_parallel_run(plan, output, .Rows_Forward)
		r2r_2d_parallel_run(plan, output, .Cols_Forward)
		return .None
	}
	if err := r2r_2d_apply_rows_with_plan(&plan.row_plan, plan.rows, plan.cols, output, false); err != .None {
		return err
	}
	return r2r_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.cols, output, plan.scratch, false)
}

r2r_2d_inverse :: proc(plan: ^R2R_Plan_2D, input: []f64, output: []f64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(input) != plan.rows*plan.cols || len(output) != plan.rows*plan.cols {
		return .Size_Mismatch
	}
	copy(output, input)
	if plan.parallel_enabled {
		r2r_2d_parallel_run(plan, output, .Cols_Inverse)
		r2r_2d_parallel_run(plan, output, .Rows_Inverse)
		return .None
	}
	if err := r2r_2d_transform_columns_with_plan(&plan.col_plan, plan.rows, plan.cols, output, plan.scratch, true); err != .None {
		return err
	}
	return r2r_2d_apply_rows_with_plan(&plan.row_plan, plan.rows, plan.cols, output, true)
}

r2r_2d_forward_strided_with_scratch :: proc(plan: ^R2R_Plan_2D, input: []f64, input_row_stride: int, output: []f64, output_row_stride: int, scratch: []f64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}
	if err := r2r_2d_apply_rows_with_plan_strided(&plan.row_plan, plan.rows, plan.cols, input, input_row_stride, output, output_row_stride, false); err != .None {
		return err
	}
	return r2r_2d_transform_columns_with_plan_strided(&plan.col_plan, plan.rows, plan.cols, output, output_row_stride, scratch, false)
}

r2r_2d_inverse_strided_with_scratch :: proc(plan: ^R2R_Plan_2D, input: []f64, input_row_stride: int, output: []f64, output_row_stride: int, scratch: []f64) -> Error {
	if !plan.initialized {
		return .Plan_Not_Initialized
	}
	if len(scratch) < plan.rows {
		return .Size_Mismatch
	}
	if err := r2r_2d_apply_rows_with_plan_strided(&plan.row_plan, plan.rows, plan.cols, input, input_row_stride, output, output_row_stride, true); err != .None {
		return err
	}
	return r2r_2d_transform_columns_with_plan_strided(&plan.col_plan, plan.rows, plan.cols, output, output_row_stride, scratch, true)
}

r2r_2d_forward_strided :: proc(plan: ^R2R_Plan_2D, input: []f64, input_row_stride: int, output: []f64, output_row_stride: int) -> Error {
	return r2r_2d_forward_strided_with_scratch(plan, input, input_row_stride, output, output_row_stride, plan.scratch)
}

r2r_2d_inverse_strided :: proc(plan: ^R2R_Plan_2D, input: []f64, input_row_stride: int, output: []f64, output_row_stride: int) -> Error {
	return r2r_2d_inverse_strided_with_scratch(plan, input, input_row_stride, output, output_row_stride, plan.scratch)
}
