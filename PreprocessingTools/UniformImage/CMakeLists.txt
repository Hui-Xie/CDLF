
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (LinuxFileDir ../../Interface3rdParty/LinuxFile)
    include_directories(${LinuxFileDir})

    set(UniformImageSrc UniformImage.cpp ${LinuxFileDir}/FileTools.cpp)
    add_executable(UniformImage ${UniformImageSrc})
    target_link_libraries(UniformImage  ${ITK_LIBRARIES} )

    set(UniformLabelSrc UniformLabel.cpp ${LinuxFileDir}/FileTools.cpp)
    add_executable(UniformLabel ${UniformLabelSrc})
    target_link_libraries(UniformLabel  ${ITK_LIBRARIES} )

    set(qSubImagesSrc qSubImages.cpp ${LinuxFileDir}/FileTools.cpp)
    add_executable(qSubImages ${qSubImagesSrc})

else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)