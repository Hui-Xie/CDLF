
 set (FileToolsDir ../FileTools)
 include_directories(${FileToolsDir})

 set(SplitTrainSetSrc splitTrainSet.cpp ${FileToolsDir}/FileTools.cpp)
 add_executable(SplitTrainSet ${SplitTrainSetSrc})
 target_link_libraries(SplitTrainSet )