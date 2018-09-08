//
// Created by hxie1 on 9/8/18.
//

#ifndef CDLF_FRAMEWORK_FILETOOLS_H
#define CDLF_FRAMEWORK_FILETOOLS_H

#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;


bool isExceptionFile(const string file, const vector<string> exceptionFiles);

string getUniformPathFileName(const string& inputFile);

string getDirFromFileName(const string& fullPathFileName);

bool dirExist(const string& dirPath);


#endif //CDLF_FRAMEWORK_FILETOOLS_H
