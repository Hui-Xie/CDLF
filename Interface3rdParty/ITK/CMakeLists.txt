#set(ITK_DIR D:/ITK_build)
#Or you need to config ITK_DIR in your evironment setting in clion IDE.
#    -DITK_DIR=theFullPathOfITK-BuildDir
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
