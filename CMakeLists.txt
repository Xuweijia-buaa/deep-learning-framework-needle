cmake_minimum_required(VERSION 3.2)
#project(needle C CXX)    # 3.9以前的
project(needle C CXX CUDA)  # 我们重写的。  指定编译项目，需要编译的编程语言
                            #             默认支持的语言是C, CXX (i.e. C++)。  也支持 CSharp (i.e. C#), CUDA等  3.8以后开始支持cuda

# 1 导入python环境.  pybind11需要调用python解释器中函数,需首先导入python相关的库

# 1.1生成一个子进程。执行对应命令。没有额外中间进程。 执行后打印的变量，赋给OUTPUT_VARIABLE参数中指定的变量
#    这里通过执行命令得到的变量Python_ROOT_DIR，是之后查找python的根路径。也可以自己手动设置:set(Python_ROOT_DIR "xxx")
execute_process(COMMAND python3-config --prefix                   # commond 本身： python3-config --prefix：打印当前python基路径： 如/home/xuweijia/.conda/envs/pytorch1.9 
  OUTPUT_VARIABLE Python_ROOT_DIR)                                # OUTPUT_VARIABLE XXX： 把子命令的输出，赋给变量Python_ROOT_DIR。

 # 1.2 添加项目需要的，已经放在某些位置的依赖包(或包中组件)，并把依赖包的细节加载进来。一般是已经编译好的（ FetchContent¶命令可以先下载未编译好的依赖包。一同编译。）
 #    会查找默认位置（以及指定位置），把找到的包放在结果变量中，以供之后使用
 #    查找时有2种模式：不指定时，默认先按Module mode找，再按config mode找： 
 #    find_package(Qt5Gui 5.1.0 CONFIG|MODULE REQUIRED)。
 #    Config mode：找的是包本身提供的文件，直接使用包本身提供的config,version等文件，来提供关于这个包的细节信息。找的是.cmake为后缀的文件：<PackageName>Config.cmake。 更可靠：
 #                 第三方包可以提供config files，定义CMake的targets, variables, commands，安装时放在lib/cmake/<PackageName>下。被find_packages()搜索。
 #                 一般config files文件名叫<PackageName>Config.cmake 或 <LowercasePackageName>-config.cmake，包含该包被安装在系统中的位置，头文件的位置等。
 #                 生成一个可供cmake使用的包和对应的config文件，可参考https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#creating-a-package-configuration-file
 #                 find_package对这些配置文件的搜索过程：
 #                 不同OS搜索的默认路径不同，比如unix下搜/usr/lib等，Windows下搜索Program Files   
 #                 不在默认路径中安装的包，可通过CMAKE_PREFIX_PATH变量，指定包对应上级目录(代替prefix）：
 #                 比如指定CMAKE_PREFIX_PATH=/opt/somepackage。查找配置文件：/opt/somepackage/lib/cmake/somePackage/SomePackageConfig.cmake
 #                 可参考：https://cmake.org/cmake/help/latest/command/find_package.html#search-procedure
 #    Module mode: 包本身可能不提供cmake相关的config,提供了独立的Find modules（提供包的信息）。 寻找Find<PackageName>.cmake文件 （从CMAKE_MODULE_PATH、Cmake自己安装路径）。其中CMAKE_MODULE_PATH是指定搜索路径，初始空，项目可设
 #                 其中需要寻找的FindSomePackage.cmake文件。一般不是包本身提供的。需要自己在cmake中指定CMAKE_MODULE_PATH路径，来放置这类型文件。
 #                 除非包不提供config filec，否则都应提供上边config mode模式.  写find_module: https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#id10
 #   这2种模式下，找到了的话，都会设置设置<PackageName>_FOUND。（config_mode下： 找到了config_file会设置<PackageName>_DIR变量，为configuration file文件的位置）
 #              REQUIRED：没找到，直接中断。  
 #              COMPONENTS：支持只导入包中部分组件。其中有一个没导入，就认为失败。组件由包本身定义，比如vector相对std
 #                         find_package(Qt5 5.1.0 COMPONENTS Widgets Xml Sql)
 #  如果没有特定指定COMPONENTS，都会导入包提供的目标和其他变量。
 #  导入的目标，同cmake中的普通target，其中提供的一些库，可以被链接到该project中
 #  一般提供的包，会设置对应变量如包的头文件位置，库安装位置等：${pkg_INCLUDE_DIR}; ${pkg_LIBRARY}
find_package(Python COMPONENTS Development Interpreter REQUIRED)   # 导入python对应的组件：开发环境(含include目录和库文件），python的解释器。
                                                                   # 导入的目标含python的头文件目录，设在Python_INCLUDE_DIRS变量中
                                                                   # 参考： https://cmake.org/cmake/help/latest/module/FindPython.html https://cmake.org/cmake/help/latest/module/FindPython3.html 
                                                                   #       可以直接用FindPython3命令，但分别指定可保持版本一致性. 其中Python_ROOT_DIR指定了要找的python的根目录。
                                                                   #       https://www.cnblogs.com/mxnote/p/16743186.html         

#  1.3 把python的头文件Python.h所在路径，加入cmake要找的头文件路径中。
#      Python.h是CPython中的C语言头文件，使用它可以在C语言中调用Python解释器中的函数。在编写C语言扩展模块时，需要引用此头文件
include_directories(${Python_INCLUDE_DIRS})                        # 把python的头文件Python.h所在路径，加入cmake要找的头文件路径中。之后pybind调用，可以把c++类型，转python类型。生成的.so在python中调用


# 2 find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir             # 通过run library module as a script，执行python包pybind11. 输出pybind11包的cmake配置文件位置
  RESULT_VARIABLE __pybind_exit_code                               # RESULT_VARIABLE：记录进程的返回值，到__pybind_exit_code
  OUTPUT_VARIABLE __pybind_path                                    # __pybind_path变量，存包的cmake配置文件位置：/home/xuweijia/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11
  OUTPUT_STRIP_TRAILING_WHITESPACE)                                #                     去掉可能空格
find_package(pybind11 PATHS ${__pybind_path})                      # 根据PATHS规定的指定路径，寻找pybind11的安装位置（cmake的配置文件位置）。找到后，即得到包的imported target和对应变量
                                                                   # 导入了已安装的pybind11的头文件位置：pybind11_INCLUDE_DIRS 
                                                                   #             pybind11提供的库文件位置：pybind11_LIBRARIES，可用来链接
                                                                   # 参考： https://pybind11.readthedocs.io/en/stable/cmake/index.html 


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")  # 设置编译命令.   CMAKE_CXX_FLAGS是CMake预定义的内建变量。需要在add_executable前指定，传给编译器
  set(CMAKE_CUDA_STANDARD 14)                                             # CMAKE_CUDA_STANDARD：CMake变量，用于指定使用CUDA编译器时所采用的C++标准版本。可设为14或17，前提是CUDA版本支持该标准
endif()                                                               

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})                      # 把pybind11的头文件路径，添加到此CMakeLists中每个目标的INCLUDE_DIRECTORIES属性中.是一个全局属性。使得编译时可以找到。 -Include
                                                                          # （相比于target_include_directories只为特定目标添加头文件路径）
                                                                          # 设置SYSTEM，是把该路径设置到system级别的include路径中。会跳过警告/不计算依赖。
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})                            # 把pybind11库文件，设置到LINKER_LIBS中（list变量），之后用来链接。


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)   # 把一些源码编译成一个共享库文件ndarray_backend_cpu,作为target。用MODULE模式，不链接到其他target上，但是可以被runtime动态调用。（python调用该.so，类似dlopen）.
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})     # 为该target，链接刚刚找到的pybind11提供的库。
pybind11_extension(ndarray_backend_cpu)                              # pybind11提供的函数：https://pybind11.readthedocs.io/en/stable/cmake/index.html
pybind11_strip(ndarray_backend_cpu)                                  # 设置该.so目标的拓展名，并在构建结束后strip it。


# 设置输出的target的属性。可以设置任何kv. 能设置的target properties参考：https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray   # 设置该target（库文件)的输出文件夹.  如其他类型的target，可设置其他属性，如RUNTIME_OUTPUT_DIRECTORY(可执行文件)
  CXX_VISIBILITY_PRESET "hidden"                                                       # 设置target中 c++函数的可见性
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # The name of the operating system for which CMake is to build
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)  # 链接时的命令。 -undefined dynamic_lookup 规定如果符号没找到，在runtime时查找，把符号查找推迟到运行时。
                                                                                            #              is OSX-specific。linux默认行为就是这样。
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})                  # 同样把cuda对应的库加入include,lib中
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)  # 进程 返回值。
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80。    太低，设置NVCC的编译 架构flag
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  # 3.9版本以上的CMakeLists.txt就不再需要用cuda_add_library了，像普通的c++代码一样，仍然使用add_library和add_executable就可以。
  #CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})
  #add_library(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS}) # 通过源文件构建动态库文件。没找到源文件OPTIONS(编译参数不能写到add_library里)
  add_library(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu) # 直接添加源文件，编译对应库文件。也是Module模式


  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})           # 设置对应的链接（链接pybind11,cuda对应库）
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray  # 指定该库的输出位置。(.so)
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()

