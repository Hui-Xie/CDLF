
set(Tools_Src_Dir ${CDLF_INCLUDE_DIR}/../src)

set(ConvolutionCaculator_Src ConvolutionCaculator.cpp ${Tools_Src_Dir}/Tools.cpp)
add_executable(ConvolutionCaculator ${ConvolutionCaculator_Src})

set(TransConvolutionCaculator_Src TransConvolutionCaculator.cpp ${Tools_Src_Dir}/Tools.cpp)
add_executable(TransConvolutionCaculator ${TransConvolutionCaculator_Src})