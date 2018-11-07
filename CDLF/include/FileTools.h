//
// Created by Hui Xie on 9/8/18.
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

string getUniformPathFileName(const string& inputFile, const string& dirSuffix);

string getFileName(const string& fullPathFileName);

string getDirFromFileName(const string& fullPathFileName);

bool dirExist(const string& dirPath);

void createDir(const string& dirPath);

void getFileVector(const string& dir, vector<string>& fileVector);

void copyFile(const string& srcFilename, const string& dstFilename);

// return >2 is not empty directory.
int countEntriesInDir(const string& dirStr);

bool isEmptyDir(const string& dirStr);

#endif //CDLF_FRAMEWORK_FILETOOLS_H
