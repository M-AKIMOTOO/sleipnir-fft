ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CONFIG_MK := $(ROOT_DIR)/build/config.mk

-include $(CONFIG_MK)

PREFIX ?= /usr/local
ODIN_PATH ?= $(shell command -v odin 2>/dev/null)
ODIN ?= $(or $(ODIN_PATH),odin)
BUILD_DIR ?= $(ROOT_DIR)/build
LIB_MODE ?= static
LIB_NAME ?= sleipnir_fft
ODIN_OPT ?= speed
MICROARCH ?= native
PKG_NAME ?= sleipnir-fft
INSTALL_PKG_ROOT ?= $(PREFIX)/share/$(PKG_NAME)
INSTALL_LIB_DIR ?= $(PREFIX)/lib

ifeq ($(LIB_MODE),static)
  LIB_FILENAME := lib$(LIB_NAME).a
else ifeq ($(LIB_MODE),shared)
  LIB_FILENAME := lib$(LIB_NAME).so
else ifeq ($(LIB_MODE),dll)
  LIB_FILENAME := lib$(LIB_NAME).dll
else
  LIB_FILENAME := lib$(LIB_NAME)
endif

ifeq ($(ODIN_OPT),none)
  ODIN_OPT_FLAG :=
else
  ODIN_OPT_FLAG := -o:$(ODIN_OPT)
endif

ifeq ($(MICROARCH),none)
  MICROARCH_FLAG :=
else
  MICROARCH_FLAG := -microarch:$(MICROARCH)
endif

LIB_OUT := $(BUILD_DIR)/$(LIB_FILENAME)
ODIN_BUILD_FLAGS := $(ODIN_OPT_FLAG) $(MICROARCH_FLAG)

.PHONY: help configure check-odin build test bench clean install install-src install-lib uninstall print-config

help:
	@echo "Targets:"
	@echo "  configure   - run ./configure (use: make configure CONF_ARGS='...')"
	@echo "  build       - build FFT library binary"
	@echo "  test        - run odin tests for package src"
	@echo "  bench       - run benchmark compare script"
	@echo "  install     - install source package and built library"
	@echo "  install-src - install source package only"
	@echo "  install-lib - install built library only"
	@echo "  uninstall   - remove installed files from PREFIX/DESTDIR"
	@echo "  clean       - remove build artifacts"
	@echo "  print-config- show current configuration"
	@echo
	@echo "Current:"
	@$(MAKE) --no-print-directory print-config

configure:
	@./configure $(CONF_ARGS)

check-odin:
	@if ! command -v "$(ODIN)" >/dev/null 2>&1; then \
		echo "error: Odin compiler not found: $(ODIN)"; \
		echo "hint: set ODIN=/path/to/odin or run ./configure --odin=/path/to/odin"; \
		exit 1; \
	fi

build: check-odin
	@mkdir -p "$(BUILD_DIR)"
	@"$(ODIN)" build "$(ROOT_DIR)/src" -build-mode:$(LIB_MODE) -out:"$(LIB_OUT)" -collection:sleipnirfft="$(ROOT_DIR)" $(ODIN_BUILD_FLAGS)
	@echo "built: $(LIB_OUT)"

test: check-odin
	@"$(ODIN)" test "$(ROOT_DIR)/src" -collection:sleipnirfft="$(ROOT_DIR)" $(ODIN_BUILD_FLAGS)

bench:
	@"$(ROOT_DIR)/../scripts/bench_fft_compare.sh"

install: install-src install-lib

install-src:
	@mkdir -p "$(DESTDIR)$(INSTALL_PKG_ROOT)/src"
	@cp -f "$(ROOT_DIR)/README.md" "$(DESTDIR)$(INSTALL_PKG_ROOT)/README.md"
	@cp -f "$(ROOT_DIR)/src/README.md" "$(DESTDIR)$(INSTALL_PKG_ROOT)/src/README.md"
	@cp -f "$(ROOT_DIR)/src/"*.odin "$(DESTDIR)$(INSTALL_PKG_ROOT)/src/"
	@echo "installed source package: $(DESTDIR)$(INSTALL_PKG_ROOT)"

install-lib: build
	@mkdir -p "$(DESTDIR)$(INSTALL_LIB_DIR)"
	@cp -f "$(LIB_OUT)" "$(DESTDIR)$(INSTALL_LIB_DIR)/$(LIB_FILENAME)"
	@echo "installed library: $(DESTDIR)$(INSTALL_LIB_DIR)/$(LIB_FILENAME)"

uninstall:
	@rm -rf "$(DESTDIR)$(INSTALL_PKG_ROOT)"
	@rm -f "$(DESTDIR)$(INSTALL_LIB_DIR)/$(LIB_FILENAME)"
	@echo "removed: $(DESTDIR)$(INSTALL_PKG_ROOT)"
	@echo "removed: $(DESTDIR)$(INSTALL_LIB_DIR)/$(LIB_FILENAME)"

clean:
	@rm -rf "$(BUILD_DIR)"
	@echo "cleaned: $(BUILD_DIR)"

print-config:
	@echo "PREFIX=$(PREFIX)"
	@echo "ODIN=$(ODIN)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@echo "LIB_MODE=$(LIB_MODE)"
	@echo "LIB_NAME=$(LIB_NAME)"
	@echo "LIB_OUT=$(LIB_OUT)"
	@echo "ODIN_OPT=$(ODIN_OPT)"
	@echo "MICROARCH=$(MICROARCH)"
	@echo "PKG_NAME=$(PKG_NAME)"
	@echo "INSTALL_PKG_ROOT=$(INSTALL_PKG_ROOT)"
	@echo "INSTALL_LIB_DIR=$(INSTALL_LIB_DIR)"
