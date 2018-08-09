//
// Created by Hui Xie on 7/16/2018.
//

#ifndef RL_NONCONVEX_TOOLS_H
#define RL_NONCONVEX_TOOLS_H
#include <vector>
#include <string>

using namespace std;

bool sameVector(const vector<long>& vec1, const vector<long>& vec2);

vector<long> reverseVector(const vector<long>& src);

string vector2String(const vector<long>& src);

vector<long> operator+ (const vector<long>& left, const int offset);

vector<long> operator+ (const vector<long>& left, const vector<long>& right);

vector<long> operator- (const vector<long>& minuend, const int subtrahend);

vector<long> operator* (const vector<long>& left, const int factor);

vector<long> operator/ (const vector<long>& left, const int divisor);

void deleteOnes(vector<long>& vec);

void printVector(const vector<long>& vec);

string vector2Str(const vector<long>& vec);


#endif //RL_NONCONVEX_TOOLS_H
