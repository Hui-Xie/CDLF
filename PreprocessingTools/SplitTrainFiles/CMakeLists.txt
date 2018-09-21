
  set (LinuxFileDir ../../Interface3rdParty/LinuxFile)
  include_directories(${LinuxFileDir})

 set(SplitTrainSetSrc splitTrainSet.cpp ${LinuxFileDir}/FileTools.cpp)
 add_executable(SplitTrainSet ${SplitTrainSetSrc})
 target_link_libraries(SplitTrainSet )