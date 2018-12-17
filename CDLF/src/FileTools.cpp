//
// Created by Hui Xie on 9/8/18.
//

#include "FileTools.h"
#include <fstream>
#include <dirent.h>
#include <sys/types.h>

bool isExceptionFile(const string file, const vector<string> exceptionFiles){
    int N = exceptionFiles.size();
    for(int i =0; i< N; ++i){
        if (file == exceptionFiles[i]) return true;
    }
    return false;
}

string getUniformPathFileName(const string& inputFile, const string& dirSuffix){
    string result = inputFile;
    size_t pos = result.rfind('/');
    if (pos != string::npos){
        result.insert(pos, dirSuffix);
    }
    else{
        result = "";
    }
    return result;
}

string getFileName(const string& fullPathFileName){
    string result = "";
    size_t pos = fullPathFileName.rfind('/');
    if (pos != string::npos){
        result = fullPathFileName.substr(pos+1);
    }
    else{
        result = "";
    }
    return result;

}

string getDirFromFileName(const string& fullPathFileName){
    string result = "";
    size_t pos = fullPathFileName.rfind('/');
    if (pos != string::npos){
        result = fullPathFileName.substr(0, pos);
    }
    else{
        result = "";
    }
    return result;
}

bool dirExist(const string& dirPath){
    struct stat statBuff;
    if (stat(dirPath.c_str(),&statBuff) == -1){
        return false;
    }
    if (S_IFDIR == (statBuff.st_mode & S_IFMT)){
        return true;
    }
    else{
        return false;
    }
}

void createDir(const string& dirPath){
    if (!dirExist(dirPath)){
        mkdir(dirPath.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
    }
}

void getFileVector(const string& dir, vector<string>& fileVector){
    DIR* pDir = opendir(dir.c_str());
    struct dirent* pEntry;
    while ((pEntry = readdir(pDir)) != NULL) {
        if (pEntry->d_type == DT_REG && '.' != pEntry->d_name[0]){
            fileVector.push_back(dir+"/"+pEntry->d_name);
        }
    }
    closedir(pDir);
}

void copyFile(const string& srcFilename, const string& dstFilename){
    std::ifstream  src(srcFilename.c_str(), std::ios::binary);
    std::ofstream  dst(dstFilename.c_str(),   std::ios::binary);
    dst << src.rdbuf();
}



int countEntriesInDir(const string& dirStr)
{
    int count=0;
    dirent* entity;
    DIR* pDir = opendir(dirStr.c_str());
    if (pDir == NULL) return 0;
    while((entity = readdir(pDir))!=NULL) count++;
    closedir(pDir);
    return count;
}

bool isEmptyDir(const string& dirStr){
    if (countEntriesInDir(dirStr) > 2 ) return false;
    else return true;
}
