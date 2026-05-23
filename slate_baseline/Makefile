# Build SLATE RSVD naive baseline via pkg-config.
# Requires: pkg-config slate

-include make.inc

LOCAL_SLATE_PREFIX ?= $(abspath ../third_party/slate-install)
LOCAL_SLATE_LIBDIR  ?= $(if $(wildcard $(LOCAL_SLATE_PREFIX)/lib64),$(LOCAL_SLATE_PREFIX)/lib64,$(LOCAL_SLATE_PREFIX)/lib)
ifneq ($(wildcard $(LOCAL_SLATE_PREFIX)/lib/pkgconfig/slate.pc),)
    export PKG_CONFIG_PATH := $(LOCAL_SLATE_PREFIX)/lib/pkgconfig:$(PKG_CONFIG_PATH)
endif
ifneq ($(wildcard $(LOCAL_SLATE_PREFIX)/lib64/pkgconfig/slate.pc),)
    export PKG_CONFIG_PATH := $(LOCAL_SLATE_PREFIX)/lib64/pkgconfig:$(PKG_CONFIG_PATH)
endif

SKIP_SLATE_CHECK := $(filter doctor clean,$(MAKECMDGOALS))

ifeq ($(SKIP_SLATE_CHECK),)
    pkg_exists := ${shell pkg-config --exists slate; echo $$?}
    local_slate_exists := ${shell test -f "$(LOCAL_SLATE_LIBDIR)/libslate.so"; echo $$?}

    ifeq (${pkg_exists},0)
        CXX            = ${shell pkg-config --variable=CXX slate}
        CXXFLAGS      += -O3 -std=c++17 ${shell pkg-config --cflags slate}
        LDFLAGS       += ${shell pkg-config --libs-only-L slate} ${shell pkg-config --libs-only-other slate}
        SLATE_LIBS     = ${shell pkg-config --libs-only-l slate}
    else ifeq (${local_slate_exists},0)
        CXX            = mpicxx
        CXXFLAGS      += -O3 -std=c++17 -fopenmp -I$(LOCAL_SLATE_PREFIX)/include
        LDFLAGS       += -fopenmp -L$(LOCAL_SLATE_LIBDIR) -Wl,-rpath,$(LOCAL_SLATE_LIBDIR)
        SLATE_LIBS     = -lslate -llapackpp -lblaspp -lcudart -lcublas -lcusolver -lcublasLt
    else
        $(error Could not find SLATE via pkg-config or local install at $(LOCAL_SLATE_LIBDIR). Try: source ./env_taiwania2.sh && ./install_slate_local.sh)
    endif
endif
MPIRUN         ?= $(shell command -v mpirun)

TARGET = slate_rsvd_naive_baseline
SRC    = slate_rsvd_naive_baseline.cc
OBJ    = ${SRC:.cc=.o}

.PHONY: all clean echo doctor run16

all: ${TARGET}

${TARGET}: ${OBJ}
	${CXX} -o $@ $^ ${LDFLAGS} ${SLATE_LIBS} ${LIBS}

%.o: %.cc
	${CXX} ${CXXFLAGS} -MMD -MP -c -o $@ $<

clean:
	rm -f ${TARGET} ${OBJ} *.d

echo:
	@echo "CXX=${CXX}"
	@echo "CXXFLAGS=${CXXFLAGS}"
	@echo "LDFLAGS=${LDFLAGS}"
	@echo "SLATE_LIBS=${SLATE_LIBS}"
	@echo "LOCAL_SLATE_PREFIX=${LOCAL_SLATE_PREFIX}"
	@echo "LOCAL_SLATE_LIBDIR=${LOCAL_SLATE_LIBDIR}"
	@echo "PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"

doctor:
	@echo "mpirun: $$(command -v mpirun || echo missing)"
	@echo "pkg-config: $$(command -v pkg-config || echo missing)"
	@echo "slate.pc visible: $$(pkg-config --exists slate && echo yes || echo no)"
	@echo "PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"

run16: ${TARGET}
	@if [ -z "${MPIRUN}" ]; then \
		echo "mpirun not found. Run: source ./env_taiwania2.sh"; \
		exit 1; \
	fi
	${MPIRUN} -np 16 ./${TARGET} --m 65536 --n 4096 --k 128 --oversample 32 --seed 12345

-include *.d
