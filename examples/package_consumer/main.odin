package main

import "core:fmt"
import fft "sleipnirfft:src"

main :: proc() {
	n := 8
	plan: fft.C2C_Plan
	if fft.c2c_plan_init(&plan, n) != .None {
		fmt.eprintln("failed to initialize fft plan")
		return
	}
	defer fft.c2c_plan_destroy(&plan)

	data := []complex128{
		1, 2, 3, 4, 0, 0, 0, 0,
	}
	if fft.c2c_forward_in_place(&plan, data) != .None {
		fmt.eprintln("forward fft failed")
		return
	}

	fmt.printf("sleipnir-fft %s\n", fft.VERSION_STRING)
	fmt.println(data)
}
