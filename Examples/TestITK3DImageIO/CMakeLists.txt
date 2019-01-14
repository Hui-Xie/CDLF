
message("Config ITK Notes:")
message("0  Courtesy of Kitware Company (https://itk.org/itkindex.html) for ITK library;")
message("1  download code from https://itk.org/ITK/resources/software.html into your intall directory;")
message("2  decompress, ccmake, make ITK version using same version compilers with CDLF; Different compiler will result link errors;")
# make sure ITK and CDLF to use same compiler to compile
message("3  in Clion File/Setting/Build/CMake/CMakeOptions config -DITK_DIR=the/Full/Path/Of/ITK-Build/Dir ;")
message("*************************************")


#set(ITK_DIR /Users/hxie1/Projects/ITK_build)

find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (ITKImageIODir ../../Interface3rdParty/ITK)
    include_directories(${ITKImageIODir})

    set(TestITK3DImageIOSRCS TestITK3DImageIO.cpp TestITKDataMgr.cpp ${ITKImageIODir}/ITKDataManager.cpp)
    add_executable(TestITK3DImageIO ${TestITK3DImageIOSRCS})
    target_link_libraries(TestITK3DImageIO  ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES})
else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)
