# 定义伪目标，防止与同名文件冲突
.PHONY: all build install python-install clean

# 默认执行的目标
all: build install python-install

# 1. 编译 C++ 代码
build:
	xmake

# 2. 安装共享库 (根据你的 xmake.lua，这通常是拷贝到本地或系统路径)
install: 
	xmake install

# 3. 安装 Python 包 (使用 -e 模式建议见下文)
python-install: 
	pip install -e ./python/

# 运行目录下所有的算子测试
test-all: all
	@for file in $(shell ls test/ops/*.py); do \
		echo "Running $$file ..."; \
		python3 $$file --device cpu; \
	done
# 清理编译
clean:
	xmake clean
	rm -rf python/*.egg-info