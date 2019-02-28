//
// Created by Hui Xie on 7/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_TOOLS_H
#define RL_NONCONVEX_TOOLS_H
#include <vector>
#include <string>

using namespace std;

#define printCurrentFileLine()   {std::cerr << "Execute at line " << __LINE__ << " of file: "<< __FILE__<<std::endl;}


bool sameVector(const vector<int>& vec1, const vector<int>& vec2);

size_t length(const vector<int>& vec);

size_t length(const int N, const int array[]);

bool sameLength(const vector<int>& vec1, const vector<int>& vec2);

vector<int> reverseVector(const vector<int>& src);

vector<float> reverseVector(const vector<float>& src);

vector<int> operator+ (const vector<int>& left, const int offset);

vector<int> operator+ (const vector<int>& left, const vector<int>& right);

vector<int> operator- (const vector<int>& left, const vector<int>& right);

vector<int> operator- (const vector<int>& minuend, const int subtrahend);

vector<int> operator* (const vector<int>& left, const int factor);

vector<int> operator* (const vector<int>& left, const vector<int>& right);

vector<int> operator/ (const vector<int>& left, const int divisor);

bool operator<= (const vector<int>& left, const vector<int>& right);

bool operator!= (const vector<int>& left, const vector<int>& right);

bool operator== (const vector<int>& left, const vector<int>& right);

void deleteOnes(vector<int>& vec);

int maxElement(const vector<int>& vec);

vector<int> nonZeroIndex(const vector<int>& vec);

void printVector(const vector<int>& vec);

string vector2Str(const vector<int>& vec);

string array2Str(const int array[], const int  N);

//users needs to delete[] array after use.
void vector2Array(const vector<int>& vec, int & N, int array[]);

vector<int> str2Vector(const string& str);

vector<int> generateRandomSequence(const int range);

vector<int> generatePositiveNegativeRandomNumber(const int vectorSize, const int maxValue);

vector<float> generatePositiveNegativeRandomRadian(const int vectorSize, const float maxRadian);

string getStemName(const string& filename);

void printCurrentLocalTime();

string getCurTimeStr();

vector<int> genDimsSpan(const vector<int> vec);

string eraseAllSpaces(string str);

void dimA2SpanA(const int* dimA, const int N, int * spanA);

bool isElementBiggerThan0(const vector<int>& vec);
bool isElementEqual1(const vector<int>& vec);
bool isElementEqual0(const vector<float>& vec);

bool isContainSubstr(const string& str, const string subStr);

// radianVec indicating rotating angle about axis 0,1,2;  R[3][4] are rotation matrix without translation,
void getRotationMatrix(const vector<float> radianVec, double R[3][4]);

// After Hinge rotation, translate the object into first quadrant with all index >0
vector<int> getRotatedDims_UpdateTranslation(const vector<int> dims, double R[3][4]);

vector<int> getIppRotatedDims_UpdateTranslation(const vector<int> ippDims, double ippR[3][4]);

void randomTranslate(vector<int>& vec, const int translationMaxValue);

vector<int> getTopLeftIndexFrom(const vector<int> &imageDims, const vector<int> &subImageDims,
                                const vector<int>&  center);
#endif //RL_NONCONVEX_TOOLS_H
