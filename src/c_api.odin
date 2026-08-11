package fft

import "core:mem"
import "base:runtime"

C_API_MAX_LENGTH :: i64(1 << 30)

c_api_valid_length :: #force_inline proc "contextless" (n: i64) -> bool {
	return n > 0 && n <= C_API_MAX_LENGTH
}

c_api_f32_slice :: #force_inline proc(data: rawptr, n: i64) -> []complex64 {
	return mem.slice_ptr(cast(^complex64)data, int(n))
}

c_api_f64_slice :: #force_inline proc(data: rawptr, n: i64) -> []complex128 {
	return mem.slice_ptr(cast(^complex128)data, int(n))
}

@(export, link_name="sleipnir_fft_f32_plan_create")
sleipnir_fft_f32_plan_create :: proc "c" (n: i64) -> rawptr {
	context = runtime.default_context()
	if !c_api_valid_length(n) {
		return nil
	}

	plan := new(C2C_Plan_F32)
	if plan == nil {
		return nil
	}
	if err := c2c_plan_init_f32(plan, int(n)); err != .None {
		free(rawptr(plan))
		return nil
	}
	return rawptr(plan)
}

@(export, link_name="sleipnir_fft_f32_plan_destroy")
sleipnir_fft_f32_plan_destroy :: proc "c" (handle: rawptr) {
	context = runtime.default_context()
	if handle == nil {
		return
	}
	plan := cast(^C2C_Plan_F32)handle
	c2c_plan_destroy_f32(plan)
	free(rawptr(plan))
}

@(export, link_name="sleipnir_fft_f32_forward")
sleipnir_fft_f32_forward :: proc "c" (handle, data: rawptr, n: i64) -> i32 {
	context = runtime.default_context()
	if handle == nil || data == nil || !c_api_valid_length(n) {
		return i32(Error.Invalid_Length)
	}
	plan := cast(^C2C_Plan_F32)handle
	return i32(c2c_forward_in_place_f32(plan, c_api_f32_slice(data, n)))
}

@(export, link_name="sleipnir_fft_f32_inverse")
sleipnir_fft_f32_inverse :: proc "c" (handle, data: rawptr, n: i64) -> i32 {
	context = runtime.default_context()
	if handle == nil || data == nil || !c_api_valid_length(n) {
		return i32(Error.Invalid_Length)
	}
	plan := cast(^C2C_Plan_F32)handle
	return i32(c2c_inverse_in_place_f32(plan, c_api_f32_slice(data, n)))
}

@(export, link_name="sleipnir_fft_f64_plan_create")
sleipnir_fft_f64_plan_create :: proc "c" (n: i64) -> rawptr {
	context = runtime.default_context()
	if !c_api_valid_length(n) {
		return nil
	}

	plan := new(C2C_Plan)
	if plan == nil {
		return nil
	}
	if err := c2c_plan_init(plan, int(n)); err != .None {
		free(rawptr(plan))
		return nil
	}
	return rawptr(plan)
}

@(export, link_name="sleipnir_fft_f64_plan_destroy")
sleipnir_fft_f64_plan_destroy :: proc "c" (handle: rawptr) {
	context = runtime.default_context()
	if handle == nil {
		return
	}
	plan := cast(^C2C_Plan)handle
	c2c_plan_destroy(plan)
	free(rawptr(plan))
}

@(export, link_name="sleipnir_fft_f64_forward")
sleipnir_fft_f64_forward :: proc "c" (handle, data: rawptr, n: i64) -> i32 {
	context = runtime.default_context()
	if handle == nil || data == nil || !c_api_valid_length(n) {
		return i32(Error.Invalid_Length)
	}
	plan := cast(^C2C_Plan)handle
	return i32(c2c_forward_in_place(plan, c_api_f64_slice(data, n)))
}

@(export, link_name="sleipnir_fft_f64_inverse")
sleipnir_fft_f64_inverse :: proc "c" (handle, data: rawptr, n: i64) -> i32 {
	context = runtime.default_context()
	if handle == nil || data == nil || !c_api_valid_length(n) {
		return i32(Error.Invalid_Length)
	}
	plan := cast(^C2C_Plan)handle
	return i32(c2c_inverse_in_place(plan, c_api_f64_slice(data, n)))
}
