
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set(UniformImageSrc UniformImage.cpp )
    add_executable(UniformImage ${UniformImageSrc})
    target_link_libraries(UniformImage  ${ITK_LIBRARIES} ${CDLFLib})

    set(UniformLabelSrc UniformLabel.cpp )
    add_executable(UniformLabel ${UniformLabelSrc})
    target_link_libraries(UniformLabel  ${ITK_LIBRARIES} ${CDLFLib})

    set(qSubImagesSrc qSubImages.cpp )
    add_executable(qSubImages ${qSubImagesSrc})
    target_link_libraries(qSubImages  ${CDLFLib} pthread)

else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)
