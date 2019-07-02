# Collect source files to get target
# these soure files is stored into "srcs" variable
# GLOB_RECURSE递归遍历 .cpp文件，
# ${CMAKE_CURRENT_SOURCE_DIR}表示该CMakeLists.txt所在的目录， tools/
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp) 

# build each source file independently
# 每次loop, source被赋值一个文件
foreach(source ${srcs})
    # 应该是去文件名，消除后缀名
    get_filename_component(name ${source} NAME_WE)
    message(STATUS ${name})

    # caffe target already exists
    if (name MATCHES "caffe")
        set(name ${name}.bin)
    endif()

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${Caffe_LINK})
    #function(caffe_default_properties target)
    # set_target_properties(${target} PROPERTIES
    #  DEBUG_POSTFIX ${Caffe_DEBUG_POSTFIX}
    #  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    #  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    #  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    caffe_default_properties(${name}) # ?

    # set back RUNTIME_OUTPUT_DIRECTORY
    # function(caffe_set_runtime_directory target dir)
    #  set_target_properties(${target} PROPERTIES
    #   RUNTIME_OUTPUT_DIRECTORY "${dir}")
    #   endfunction()
    caffe_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
    
    # function(caffe_set_solution_folder target folder)
    #   if(USE_PROJECT_FOLDERS)
    #       # FOLDER　指定target的在它属于的文件夹下
    #   set_target_properties(${target} PROPERTIES FOLDER "${folder}")
    #   endif()
    #   endfunction()
    # 设置项目文件路径,默认状态下所有没有 FOLDER 属性的构建目标都会出现在 IDE 的顶层实体下
    # 必须要在顶层 CMakeLists.txt 中使用 set_property(GLOBAL PROPERTY USE_FOLDERS ON) 命令来开启目标层次结构，才可以组织我们的项目层次
    # 注意这个是建立target时，将每个不同子项目的target建立在它所在的子目录下，这技术需要制定target的folder属性
    caffe_set_solution_folder(${name} tools)

    # restore output name without suffix
    if (name MATCHES "caffe.bin")
        # This sets the base name for output files created for an executable or library target.
        # 设置别名 
        set_target_properties(${name} PROPERTIES OUTPUT_NAME caffe)
    endif()

    # install
    install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})

endforeach(source)