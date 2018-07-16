//
// Created by Sheen156 on 7/16/2018.
//
#include "Tools.h"


bool sameVector(const vector<int>& vec1, const vector<int>& vec2){
    if (vec1.size() != vec2.size()){
        return false;
    }
    else{
        int length = vec1.size();
        for (int i=0; i< length; ++i){
            if (vec1[i] != vec2[i]) return false;
        }
        return true;
    }
}

vector<int> reverseVector(const vector<int>& src){
    int size = src.size();
    vector<int> target;
    for(int i = size-1; i>=0; --i){
        target.push_back(src[i]);
    }
    return target;
}


