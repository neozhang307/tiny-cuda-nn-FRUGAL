all: build

.PHONY: config
config:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: build
build:
	cmake --build build --config RelWithDebInfo -j

.PHONY: build-sequentially
build-sequentially:
	cmake --build build --config RelWithDebInfo

.PHONY: build-debug
build-debug:
	cmake --build build --config Debug -j

.PHONY: clean
clean:
	rm -rf ./build

.PHONY: run
run:
	./build/mlp_learning_an_image data/images/albert.jpg tcnnConfig.json 1 inference.jpg

.PHONY: run-steps
run-steps:
	./build/mlp_learning_an_image data/images/albert.jpg tcnnConfig.json $(STEPS) $(INFERENCE)

.PHONY: update-memopt
update-memopt:
	git submodule update --remote --merge dependencies/optimize-cuda-memory-usage-v1/
