//
// Created by Sheen156 on 7/16/2018.
//

#ifndef RL_NONCONVEX_TOOLS_H
#define RL_NONCONVEX_TOOLS_H
#include <vector>
#include <string>
using namespace std;

bool sameVector(const vector<int>& vec1, const vector<int>& vec2);

vector<int> reverseVector(const vector<int>& src);

string vector2String(const vector<int>& src);

vector<int> operator+ (const vector<int>& left, const int offset);

vector<int> operator+ (const vector<int>& left, const vector<int>& right);

vector<int> operator* (const vector<int>& left, const int factor);


#endif //RL_NONCONVEX_TOOLS_H
