.PHONY: lib, pybind, clean, format, all

all: lib


lib:
    # @:关闭命令的回显
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
    # 用black工具对当前文件中的python文件格式化 用clang-format格式化c++(-i:inplace)
	python3 -m black . 
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
