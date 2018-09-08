//
// Created by hxie1 on 9/8/18.
//

#include "FileTools.h"

bool isExceptionFile(const string file, const vector<string> exceptionFiles){
    long N = exceptionFiles.size();
    for(int i =0; i< N; ++i){
        if (file == exceptionFiles[i]) return true;
    }
    return false;
}

string getUniformPathFileName(const string& inputFile){
    string result = inputFile;
    size_t pos = result.rfind('/');
    if (pos != string::npos){
        result.insert(pos, "_uniform");
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
