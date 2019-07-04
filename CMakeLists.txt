# 设定cmake最低版本。高版本cmake提供更多的功能（例如cmake3.13开始提供target_link_directories()）或解决bug（例如OpenMP的设定问题）
# 这只是最低版本要求，编译的时候会根据当前版本进行编译
cmake_minimum_required(VERSION 2.8.7)

# POLICY是策略的意思，cmake中的poilcy用来在新版本的cmake中开启、关闭老版本中逐渐被放弃的功能特性
if(POLICY CMP0046)
    message("policy CMP0046")
    cmake_policy(SET CMP0046 NEW)
endif()

if(POLICY CMP0054)
    message("policy CMP0046")
    cmake_policy(SET CMP0054 NEW)
endif()

#------------- define project------------
# C CXX 表明是C/C++工程,其实没必要写出来,因为CMake默认是开启了这两个的。
# 这句命令执行后，自动产生了5个变量：
# PROJECT_NAME，值等于Caffe
# PROJECT_SOURCE_DIR，是CMakeLists.txt所在目录，通常是项目根目录（奇葩的项目比如protobuf，把CMakeLists.txt放在cmake子目录的也有）
# PROJECT_BINARY_DIR，是执行cmake命令时所在的目录，通常是build一类的用户自行创建的目录。
# Caffe_SOURCE_DIR，此时同PROJECT_SOURCE_DIR
# Caffe_BINARY_DIR，此时同PROJECT_BINARY_DIR
project(Caffe C CXX)


# ------------------caffe version settings
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")

# add_definitions()命令通常用来添加C/C++中的宏
# add_defitions(-DCPY_ONLY) ，给编译器传递了预定义的宏CPU_ONLY，相当于代码中增加了一句#define CPU_ONLY
# add_defitions(-DMAX_PATH_LEN=256)，则相当于#define MAX_PATH_LEN 256
# 根据文档，实际上add_definitions()可以添加任意的编译器flags，只不过像添加头文件搜索路径等flags被交给include_directory()等命令了
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# --------------Using cmake scripts and modules, to find some packages
# list(APPEND VAR_NAME VAR_VALUE)这一用法，表示给变量VAR_NAME追加一个元素VAR_VALUE。虽然我写成VAR_NAME，但前面有提到，
# cmake中的变量几乎都是字符串或字符串数组，这里VAR_NAME你就当它是一个数组就好了，而当后续使用${VAR_NAME}时输出的是”整个数组的值“。
# 具体的说，这里是把项目根目录(CMakeLists.txt在项目根目录，${PROJECT_SOURCE_DIR}表示CMakeLists.txt所在目录）下的cmake/Modules子目录对应的路径值，
# 追加到CMAKE_MODULE_PATH中；CMAKE_MODULE_PATH后续可能被include()和find_package()等命令所使用。
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# include 两种作用，包含模块，包含文件
# 所谓包含文件，例如include(utils.cmake)，把当前路径下的utils.cmake包含进来，基本等同于C/C++中的#include指令。通常，include文件的话文件应该是带有后缀名的。
# 所谓包含模块，比如include(xxx)，是说在CMAKE_MODULE_PATH变量对应的目录，或者CMake安装包自带的Modules目录（比如mac下brew装的cmake对应的是/usr/local/share/cmake/Modules)里面寻找xxx.cmake文件

# 包含模块
# 这里是把CMake安装包提供的ExternalProject.cmake(例如我的是/usr/local/share/cmake/Modules/ExternalProject.cmake）文件包含进来
include(ExternalProject)
include(GNUInstallDirs)

# 包含文件
# 定义了一些通用的（适用于其他项目的）函数和宏，用于变量（数组）的打印、合并、去重、比较等
# （吐槽：cmake语法比较奇葩，相当一段时间之后我才发现它是lisp方式的语法，也就是函数（命令）是一等公民）
include(cmake/Utils.cmake)
# 定义了Caffe项目本身的一些函数和宏，例如源码文件组织、目录组织等
include(cmake/Targets.cmake)
# 杂项，比较抠细节的一些设定，比如通常CMAKE_BUILD_TYPE基本够用了，但是这里通过CMAKE_CONFIGURATION_TYPES来辅助设定CMAKE_BUILD_TYPE，等
include(cmake/Misc.cmake)
# 定义了4个打印函数，用来打印Caffe的一些信息，执行CMake时会在终端输出，相比于散落在各个地方的message()语句会更加系统一些
include(cmake/Summary.cmake)
# 整个caffe编译好之后，如果别的项目要用它，那它也应该用cmake脚本提供配置信息。生成一些脚本配置信息，供其他项目使用
include(cmake/ConfigGen.cmake)

# ---------------------设置外部控置参数 OPtions
# 此处在cmake/Utils.cmake中，将option封装成了caffe_option,　用于设置外部控制参数, 一般option是没有条件判断的，因此封装成带有条件判断的caffe_option
# 其中，option()指令设定的东西也可以被用CMake -Dxxx=ON的形式来覆盖。
# 这里是设定各种option，也就是”开关“，然后后续根据开关的取值（布尔类型的变量，利用if和else来判断），编写各自的构建规则。
# 其中caffe_option()是cmake/Utils.cmake中定义的，它相比于cmake自带的option()命令，增加了可选的条件控制字段

caffe_option(CPU_ONLY "Build caffe without cuda support" OFF) # TODO rename to USE_CUDA
caffe_option(USE_CUDNN "build caffe with cudnn library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "build caffe with nccl library support" OFF)
caffe_option(BUILD_SHARED_LIBS "build shared libraries" ON)
caffe_option(BUILD_python "build python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which python version to use")
caffe_option(BUILD_matlab "build matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs "build documentation" ON IF UNIX OR APPLE)
# 通过再Dependices.cmake中控制list(APPEND Caffe_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)，生成WITH_PYTHON_LAYER宏变量
caffe_option(BUILD_python_layer "build the caffe python layer" ON)
caffe_option(USE_OPENCV "build with opencv suport" ON)
caffe_option(USE_LEVELDB "build with levelDB" ON)
caffe_option(USE_LMDB "build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "allow MDB_NOBLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "link with openMP (when your Blas wants openmp and you get linker error)"  OFF)

# This code is taken from https://github.com/sh1r0/caffe-android-lib
caffe_option(USE_HDF5 "build with hdf5" ON)

# ----------dependencies
# 这里是包含Dependencies.cmake，它里面配置了Caffe的绝大多数依赖库
# Boost, Threads, OpenMP, Google-glog, Google-gflags, Google-protobuf, HDF5, LMDB
# LevelDB, Snappy, CUDA, OpenCV, BLAS, Python, Matlab ,Doxygen
# 每一个依赖库库都直接（在Dependencies.cmake中）或间接（在各自的cmake脚本文件中）使用find_package()命令来查找包

# find_package(Xxx)如果执行成功，则提供相应的Xxx_INCLUDE_DIR、Xxx_LIBRARY_DIR等变量，看起来挺方便，
# 但其实并不是所有的库都提供了同样的变量后缀，其实都是由库的官方作者或第三方提供的xxx.cmake等脚本来得到的，依赖于生态。
# find_packge(Xxx)实际中往往是翻车重灾区。它其实有N大查找顺序，而CSDN上的博客中往往就瞎弄一个，你照搬后还是不行。具体例子：
### 系统包管理工具装的OpenCV不带contrib模块，想使用自行编译的OpenCV但是git clone下来的开源代码执行后找不到自己编译的OpenCV。其实只要知道N大查找顺序，设定CMAKE_PREFIX_PATH中包含OpenCV路径后基本都能找到。
### Caffe基于cmake编译，依赖于Boost，系统里用apt或brew装了Boost，同时也自行编译了高版本Boost，现在Caffe编译时cmake只认自行编译版的Boost，指定N大查找顺序也不能找到系统的Boost。切换已安装的多个Boost给CMake find_package()，这时候需要看看FindBoost.cmake是怎么写的，必须提供它里面说的字样的变量（表示include和lib的查找路径），才能让find_package()起作用。
### CMake编译安装了多个版本的Caffe（比如官方Caffe、SSD的Caffe），~/.cmake目录下会缓存一个caffe，而现在手头有一个做人脸检测的工程依赖Caffe，而你希望它用官方Caffe而不是SSD-Caffe，这个缓存目录很可能捣乱，这个我认为是某些项目比如Caffe的export输出是多余的，反而容易造成混淆。
# 只需要注意如果某个包你安装了但是cmake却没有找到，那就需要在find_package()前进行设定，以及之后排
include(cmake/Dependencies.cmake) # 有protobuf的设置

# ---------flags
# 通过设定CMAKE_CXX_FLAGS，cmake生成各自平台的makefile、.sln或xcodeproject文件时设定同样的CXXFLAGS给编译器。
# 如果是.c文件，则由c编译器编译，对应的是CMAKE_C_FLAGS
# -fPIC作用于编译阶段，告诉编译器产生与位置无关代码(Position-Independent Code)，则产生的代码中，没有绝对地址，全部使用相对地址，故而代码可以被加载器加载到内存的任意位置，都可以正确的执行。这正是共享库所要求的，共享库被加载时，在内存的位置不是固定的。
# -Wall则是开启所有警告。根据个人的开发经验，C编译器的警告不能完全忽视，有些wanring其实应当当做error来对待，

if (UNIX OR APPLE)
    # -fPIC表示使用位置无关（相对位置）编译程序地址，-Wall　表示编译后输出所有警告
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" -fPIC -Wall)
    # list(APPEND CMAKE_CXX_FLAGS "-fPIC -Wall")
endif()

# 这里是设置Caffe_LINK这一变量，后续链接阶段会用到。它定义在cmake/Targets.cmake中
# 可以看到，如果是编共享库（动态库），则就叫caffe；否则，则增加一些链接器的flags：
# -Wl是告诉编译器，后面紧跟的是链接器的flags而不是编译器的flags（现在的编译器往往是包含了调用连接器的步骤）
# 连接器ｆｌａｇｓ:https://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_node/ld_3.html
caffe_set_caffe_link()

# 在前面已经include(cmake/Dependencies.cmake)的情况下，Dependencies.cmake中的include(cmake/Cuda.cmake)使得Cuda的设定也被载入。
# 而Cuda.cmake中的最后，判断如果当前操作系统是苹果系统并且>10.8、cuda版本小于7.0，那么使用libstdc++而不是libc+
# 需注意：　通常执行cmake后最前面会输出它所使用的C、C++编译器的可执行文件完整路径，然后一个同事的机器上把CXX环境变量设为/usr/bin/gcc，
# 导致编译.cpp文件时是用CXX这一环境变量——也就是gcc——来编译.cpp文件。编译.cpp，如果是C++编译器来编译，链接阶段默认会把标准库链接进去，而现在是C编译器，没有明确指出要链接C++标准库，就会导致链接出问题，
# 虽然他的CMakeLists.txt中曾经加入过libstdc++库，但是显然这很容易翻车，CXX环境变量不应该设定为/usr/bin/gcc; C编译器ＧＣＣ编译.cpp会出现链接错误
if (USE_libstdcpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    message("-- Warning: forcing libstdc++ (controlled by USE_libdtdcpp option in cmake")
endif()


# ---------------- warning 
# 这里添加的编译器flags，是用来屏蔽特定类型的警告的。虽说眼不见心不烦，关掉后少些warning输出
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# -------------------- config generation
# 将caffe_config.h.in中的一些变量或者宏，生成到caffe_config.h
### /* Sources directory */
### define SOURCE_FOLDER "${PROJECT_SOURCE_DIR}"
### /* Binaries directory */
### define BINARY_FOLDER "${PROJECT_BINARY_DIR}"
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# --------- includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PEOJECT_BINARY_DIR})

# -----------includes and defines for cuda
# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
# 擦亮眼睛：Caffe的cmake脚本中分别定义了Caffe_INCLUDE_DIR和Caffe_INCLUDE_DIRS两个变量，只相差一个S，稍不留神容易混掉：
### 不带S的值是$Caffe_ROOT/include，
### 带S的值是各个依赖库的头文件搜索路径（在Dependencies.cmake中多次list(APPEND得到的。类似的，Caffe_DEFINITIONS也是在Dependencies.cmake中设定的。
#这里判断出如果有CUDA的话就把Caffe_INCLUDE_DIRS变量中的PUBLIC和PRIVATE都去掉，把Caffe_DEFINITIONS中的PUBLIC和PRIVATE也去掉。

if(HAVE_CUDA)
    # pass include pathes to cuda include_directories()
    set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
    list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
    cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

    # add definition to nvcc flags directly，　definitions 应该是各库的表示
    # Caffe_ALL_DEFINITIONS 在cmake/Dependencies.cmake中定义，
    ##  而且利用list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)语句累加不同的变量定义
    ## 这个只是用来标记的，并没有实际含义，到时候可以根据这个标记变量过滤不同的item，
    ## PRIVATE PUBLIC 和-DUSE_LMDB是等级别的，在使用时根据需要去掉，如第二行list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)的作用
    message(STATUS "&&&&&&&&&&&&&&& cmake definition ${Caffe_DEFINITIONS}")
    set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
    list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
    list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# -------------------sudirectories
# 使用add_subdirectory()，意思是说把子目录中的CMakeLists.txt文件加载过来执行，从这个角度看似乎等同于include()命令。
# 实则不然，因为它除了按给定目录名字后需要追加"/CMakeLists.txt"来构成完整路径外，往往都是包含一个target(类似于git中的submodule了）
# 同时还可以设定别的一些参数:
### 指定binary_dir
### 设定EXCLUDE_FROM_ALL，也就是”搞一个独立的子工程“，此时需要有project()指令，并且不被包含在生成的.sln工程的ALL目标中，需要单独构建。
# 各子目录的功能：
### gtest：　googletest的源码
### src/caffe: caffe的源码构建，因为前面做了很多操作（依赖库、路径，etc），这里写的就比较少。任务只有2个：构建一个叫做caffe的库，以及test
### tool:，这一子目录下每一个cpp文件都生成一个xxx.bin的目标，而最常用的就是caffe训练接口build/caffe这个可执行文件了
### examples 这一子目录下有cpp_classification的C++代码，以及mnist，cifar10，siamse这三个例子的数据转换的代码，这四个都是C++文件，每一个都被编译出一个可执行
### python: pycaffe接口，python/caffe/_caffe.cpp编译出动态库
### matlab: matlab接口，./+caffe/private/caffe_.cpp编译出？编译出一个定制的目标，至于是啥类型，也许是动态库吧，玩过matlab和C/C++混编的都知道，用mex编译C/C++为.mexa文件，然后matlab调用.mexa文件，其实就是动态库
### docs: 文档，doxygen、jekyll都来了，以及拷贝每一个.ipynb文件。没错，

add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
# add_subdirectory(matlab)
add_subdirectory(docs)

# ----------- linter target
# add_custom_command()能定制各种target，只要你把想要执行的shell脚本命令用cmake的语法来写就可以了，很强大
# 这里依然是定制的target，具体看来是调用scripts/cpplint.py(谷歌官方C++代码风格检查工具）来执行代码风格检查。
# （G家的C++风格有一点不太好：缩进两个空格太少了，费眼睛，强烈建议和Visual Studio保持一致，用tab并且tab宽度为4个空格）
# 所谓linter就是语法检查器，除了cpplint其实还可以用cpp_check、gcc、clang等
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ----pytest target
# add_custom_command()能定制各种target，只要你把想要执行的shell脚本命令用cmake的语法来写就可以了，很强大 (也就是调用shell命令生成自定义的target)
# 如果开启了BUILD_python开关，那么执行一个定制的target（执行pytest）。
# add_dependencies()意思是指定依赖关系，这里要求pycaffe目标完成后再执行pytest目标，
# 因为pytest需要用到pycaffe生成的caffe模块。pycaffe在前面提到的add_subdirectory(python)中被构建
if(BUILD_python)
    add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
    # add_dependencies(pytest pycaffe)
endif()

# ----- uninstall target
# 这里是添加”uninstall"这一target，具体定制的target其实就是执行cmake/Uninstall.cmake脚本。这个脚本根据cmake/Uninstall.cmake.in做变量取值替换等来生成得到
# 从Uninstall.cmake.in配置文件，生成Uninstall.cmake, 供后面Uninstall target生成时使用
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY
)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake    
)

# -------------------- configuration summary
caffe_print_configuration_summary()

#---------------------- export configs generation
# 在Caffe根目录的CMakeLists.txt的最后，是打印各种配置的总的情况，以及输出各种配置(后者其实包含了install()指令的调用)
# 包含了target的构建，其中用到export(TARGETS ...)：Export targets from the build tree for use by outside projects.

# 这个函数使用了一个命令： export(TARGETS caffe caffeproto FILE "${PROJECT_BINARY_DIR}/CaffeTargets.cmake")
# Create a file <filename> that may be included by outside projects to import targets from the current project’s build tree
# 这句命令的意思是将该工程创建的caffe,caffe_proto（在tools/CMakeLists.txt中被创建）, 暴露出，输出到CaffeTargets.cmake，　这样别的工程如果要引用这个工程的caffe库，则用find_package()去找这个cmake即可

# 也用了export(PACKAGE Caffe)，Store the current build directory in the CMake user package registry for package <name>. 
# The find_package command may consider the directory while searching for package <name>.
# 随后target被创建用命令 install(EXPORT CaffeTargets DESTINATION ${install_cmake_suffix})； 　如上所述：CaffeTargets 被export创建
# The EXPORT form is useful to help outside projects use targets built and installed by the current project;

# 初期，出现一个错误，因为caffe target 没有获取，所以导致了export caffe 出错
# caffe_generate_export_configs()
