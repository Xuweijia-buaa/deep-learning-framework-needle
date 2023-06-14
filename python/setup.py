from setuptools import setup, find_packages
from setuptools.command.install import install          # 继承原始命令，并改写
from setuptools.command.install_lib import install_lib
from pathlib import Path

# class CustomInstallLib(install_lib):  # 将包的源代码安装到安装目录中
#     # install_lib命令：作用是将包的源代码安装到安装目录中
#     # pip install命令.会自动调用install_lib命令，将包的源代码安装到安装目录中。因此不需要显式调用install_lib
#     def install(self) -> List[str]:
#         # 自定义install_lib命令：把用到的库.so，也复制到安装目录中.
#         outfiles = install_lib.install(self)
#         print("exist outfiles:",outfiles)
#
#         # TODO 查找所有.so包，复制到安装目录中
#         dll_path = [p / 'lib_lightgbm.so' for p in dll_path]
#         lib_path = [str(p) for p in dll_path if p.is_file()]
#         src = lib_path[0]                       # 编译好的lib_lightgbm.so所在的文件路径
#         dst = Path(self.install_dir) / 'lightgbm'
#         dst, _ = self.copy_file(src, str(dst))    # 复制到和python安装目录相同的文件夹
#
#         # 把目标路径，加入outfiles
#         outfiles.append(dst)
#         return outfiles
#
# class CustomInstall(install):
#     def run(self) -> None:
#         # 继承install，重写自己的run方法  （如果没有指定已经提前编译好。还需要再编译一下。参考lightgbm的python-package/setup.py）
#         # 原始的python setup.py install。安装到sitepackages
#         # 默认调用install_lib命令。把编译好的.so文件，也复制到安装目录中 （默认的install不复制.so文件）
#         install.run(self) # 原始的python setup.py install

setup(
    name = "needle",              # 最终要打包成的包名
    version = "0.1",               # 包版本
    packages = find_packages(),    # 默认搜索与setup.py同一目录中的包,被一起打到needle中
                                   #            每个包之后都可以被import
    #连同用到的so文件一起打包
    package_data={
        'needle.backend_ndarray': ['*.so'],# 键是包含.so文件的包（needle.backend_ndarray），值是该包要包含的文件的列表。['*.so']，表示所有.so文件）
    },

    # 自定义命令，继承setuptools.Command，重写 run 方法
    # cmdclass={
    #     'install_lib': CustomInstallLib,  # 执行pip install命令时调用，将包的源代码安装到安装目录(site-packages)中
    #     'install': CustomInstall,         # 安装包到系统环境中。
    # },
)


#直接安装
# .so文件将被安装到安装位置处，needle/backend_ndarray目录下
# $ python setup.py install


# 打包+安装
# 打wheel包：
# .so文件将被包含在生成的源代码分发包或二进制分发包中。当用户安装这个包时，.so文件将被安装到needle/backend_ndarray目录下，可以被ndarray.py文件import
# $ python setup.py bdist_wheel

# 安装到本地(已有的so文件，随着install_lib一起安装)：
# $ pip install dist/needle-0.1-py3-none-any.whl
# Processing ./dist/needle-0.1-py3-none-any.whl
# Installing collected packages: needle
# Successfully installed needle-0.1

