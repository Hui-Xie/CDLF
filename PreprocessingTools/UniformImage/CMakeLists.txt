
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (FileToolsDir ../FileTools)
    include_directories(${FileToolsDir})

    set(UniformImageSrc UniformImage.cpp ${FileToolsDir}/FileTools.cpp)
    add_executable(UniformImage ${UniformImageSrc})
    target_link_libraries(UniformImage  ${ITK_LIBRARIES} )

    set(UniformLabelSrc UniformLabel.cpp ${FileToolsDir}/FileTools.cpp)
    add_executable(UniformLabel ${UniformLabelSrc})
    target_link_libraries(UniformLabel  ${ITK_LIBRARIES} )


else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)