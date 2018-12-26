
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set(ImageStatSrc ImageStat.cpp )
    add_executable(ImageStat ${ImageStatSrc})
    target_link_libraries(ImageStat  ${ITK_LIBRARIES} ${CDLFLib} )
else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)