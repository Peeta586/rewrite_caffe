# rewriting-caffe recode
[TOC]
按照一步一步的来实现caffe工程

## 1. 编写CMakeLists.txt
第一步编写CMakeLists.txt文档，有两个目的：
- 为了将整个工程结构有个大概了解，
- 在编写每个模块时，能跑通整个工程

## 2. 文件功能说明
```
  #include "caffe/proto/caffe.pb.h"
```
包含caffe空间定义
如果不添加该文件的include，则会产生caffe没定义; 因为它是第一个生成的.h文件，且含有namespace caffe {}的caffe命名空间定义
## 3. caffe::string 的来源
是有namespace caffe{
  std:string
}
产生的; 这个申明在common.hpp中，
**因此开始先写common.hpp**

## 4. lint是C/C++ 强大的测试工具
lint检查C程序中潜在的错误，包括（但不限于）可疑的类型组合、未使用的变量、不可达的代码以及不可移植的代码。lint会产生一系列程序员有必要从头到尾仔细阅读的诊断信息。使用lint的好处是：1.它可以检查出被编译器漏掉的错误; 2.可以关联很多文件进行错误的检查和代码分析,具有较强大灵活性.lint可以检查的错误类型大体如下:
- 可能的空指针
- 在释放内存后使用了指向该内存的指针
- 赋值次序问题
- 拼写错误
- 被0除
- 失败的case语句(遗漏了break语句)
- 不可移植的代码(依赖了特定的机器实现)
- 宏参数没有使用圆括号
- 符号的丢失
- 异常的表达式
- 变量没有初始化
- 可疑的判断语句(例如,if(x=0))
- printf/scanf的格式检查

因此，代码中注释带有//NOLINT时， 表示That's a comment. In this case, it's a comment designed to be read by a static analysis tool to tell it to shut up about this line. 让静态分析工具不要在意这一行.

## 5. CPU_ONLY 是怎么从cmake传给C++内部的
在CaffeConfig.cmake.in中传入的, **具体怎么转化为#define CPU_ONLY 的，需要考察**
cmake/Templates/CaffeConfig.cmake.in:53:set(Caffe_CPU_ONLY @CPU_ONLY@)

ConfigGen.cmake 产生CaffeTargets.cmake, 这个文件中有：
```
set_target_properties(caffe PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "USE_LMDB;USE_LEVELDB;CPU_ONLY;USE_OPENCV"
  INTERFACE_INCLUDE_DIRECTORIES "/usr/include;/usr/local/include;/usr/local/include;/usr/include;/home/lshm/anaconda3/include;/usr/include;/usr/include;/usr/local/include;/usr/local/include/opencv;/usr/include;/usr/include/atlas;/usr/include;/home/lshm/caffe/include"
）

```
**CPU_ONLY 在这传进去的
INTERFACE_COMPILE_DEFINITIONS： 官方说明：**

List of public compile definitions requirements for a library.

Targets may populate this property to publish the compile definitions required to compile against the headers for the target. The target_compile_definitions() command populates this property with values given to the PUBLIC and INTERFACE keywords. Projects may also get and set the property directly.

When target dependencies are specified using target_link_libraries(), CMake will read this property from all target dependencies to determine the build properties of the consumer.

Contents of INTERFACE_COMPILE_DEFINITIONS may use “generator expressions” with the syntax $<...>. See the [cmake-generator-expressions(7)] manual for available expressions. See the [cmake-buildsystem(7)] -manual for more on defining buildsystem properties.

**[cmake-generator-expressions(7)]---: 也就是说这个INTERFACE_COMPILE_DEFINITIONS或者target_compile_definition的设置，会在编译的时候控制一些用宏定义为判断条件的代码是否编译编译，也就是控制条件性的定义**
Generator expressions are evaluated during build system generation to produce information specific to each build configuration.

Generator expressions are allowed in the context of many target properties, such as LINK_LIBRARIES, INCLUDE_DIRECTORIES, COMPILE_DEFINITIONS and others. They may also be used when using commands to populate those properties, such as target_link_libraries(), target_include_directories(), target_compile_definitions() and others.

**They enable conditional linking, conditional definitions used when compiling,** conditional include directories, and more. The conditions may be based on the build configuration, target properties, platform information or any other queryable information.

Generator expressions have the form $<...>. To avoid confusion, this page deviates from most of the CMake documentation in that it omits angular brackets <...> around placeholders like condition, string, target, among others.

Generator expressions can be nested, as shown in most of the examples below.

## 6. device_alternate.hpp
配置 cuda的错误检查宏，以及配置block数目和threads数目

# 错误记录

## 1. proto编译错误
```c++
[ 14%] Building CXX object src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o
c++: fatal error: no input files
compilation terminated.
/bin/sh: 1: -fPIC: not found
/bin/sh: 1: -Wall: not found
src/caffe/CMakeFiles/caffeproto.dir/build.make:74: recipe for target 'src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o' failed
make[2]: *** [src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o] Error 127
CMakeFiles/Makefile2:183: recipe for target 'src/caffe/CMakeFiles/caffeproto.dir/all' failed
make[1]: *** [src/caffe/CMakeFiles/caffeproto.dir/all] Error 2
Makefile:127: recipe for target 'all' failed
make: *** [all] Error 2

查看　prtoc --version　2.6.1，　不是版本的问题
```
根据提示可以发现：
-fPIC not found
-Wall not found
查看错误处代码：
```C++
src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o: include/caffe/proto/caffe.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lshm/Projects/rewrite_caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o"
	cd /home/lshm/Projects/rewrite_caffe/build/src/caffe && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o -c /home/lshm/Projects/rewrite_caffe/build/include/caffe/proto/caffe.pb.cc

1) 去掉-fPIC, -Wall　编译成功
２）c++ 换成g++ 也不行

通过打印发现，${CXX_FLAGS}的内容是 ";-fPIC;-Wall"，而一般这两个编译选项是不带";"分号的，
因此赋值时出错；　而CXX_FLAGS是make的系统变量，它是由CMAKE_CXX_FLAGS的值生成的．而根目录下的CMakeLists.txt的　
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" -fPIC -Wall)
对照原来的caffe的CMakeLists.txt是
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
所以修改后错误消失
```
