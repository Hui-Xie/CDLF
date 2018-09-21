
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (LinuxFileDir ../../Interface3rdParty/LinuxFile)
    include_directories(${LinuxFileDir})

    set(ImageStatSrc ImageStat.cpp ${LinuxFileDir}/FileTools.cpp)
    add_executable(ImageStat ${ImageStatSrc})
    target_link_libraries(ImageStat  ${ITK_LIBRARIES} )
else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)