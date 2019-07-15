# The option allows to include in build only selected test files and exclude all others
# Usage example:
#  cmake -DBUILD_only_tests="common,net,blob,im2col_kernel"

# CACHE STRING 定义的CaChe 变量，相当于全局变量，比normal 变量低一个级别
# 然后这个变量存储再CMakeCache.txt上
set(BUILD_only_tests "" CACHE STRING "Blank or comma-separated list of files to build without 'test_' prefix and extension")
# caffe_leave_only_selected_tests (<filelist_variable> <selected_list>)
# 
caffe_leave_only_selected_tests(test_srcs ${BUILD_only_tests})
caffe_leave_only_selected_tests(test_cuda ${BUILD_only_tests})

# For 'make runtest' target we don't need to embed test data paths to
# source files, because test target is executed in source directory
# That's why the lines below are commented. TODO: remove them

# definition needed to include CMake generated files
# add_definitions(-DCMAKE_BUILD)

# generates test_data/sample_data_list.txt.gen.cmake
#caffe_configure_testdatafile(test_data/sample_data_list.txt)

set(the_target test.testbin)
set(test_args --gtest_shuffle)

if(HAVE_CUDA)
  caffe_cuda_compile(test_cuda_objs ${test_cuda})
  list(APPEND test_srcs ${test_cuda_objs} ${test_cuda})
else()
  list(APPEND test_args --gtest_filter="-*GPU*")
endif()

# ---[ Adding test target
add_executable(${the_target} EXCLUDE_FROM_ALL ${test_srcs})
# message(STATUS "Caffe_LINK: ${Caffe_LINK}")
# Caffe_LINK is caffe, 也就是测试的函数都是再caffe中呢
target_link_libraries(${the_target} gtest ${Caffe_LINK})
caffe_default_properties(${the_target})
caffe_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test")

# ---[ Adding runtest
add_custom_target(runtest COMMAND ${the_target} ${test_args}
                          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})