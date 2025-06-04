#ifndef BLOOMFILTER
#define BLOOMFILTER

#include <iostream>
#include <bitset>
#include "datatypes.hpp"
#include"hash.h"

using namespace std;

#define hash_num 3
#define fild_sel 8

template<int width>
class BloomFilter
{
private:
    uint32_t index[hash_num];
    bitset<width> filter;
    

public:
    BloomFilter(){}
    ~BloomFilter(){}

    bool contains(unsigned char* str, int keylen, uint32_t* hval1)
    {
        MurmurHash3_x64_128(str, keylen, 800, (unsigned char*)hval1);

        for (int i = 0; i < hash_num; i++)
        {
             index[i] = hval1[i] % width;
        }

        bool ctain = true;

        for (int i = 0; i < hash_num; i++)
            ctain = ctain && filter[index[i]];
        
        if (ctain)
        {
            return ctain;
        }
        else 
        {
            for (int i = 0; i < hash_num; i++)
                filter[index[i]] = 1;
            return ctain;
        }
    }

    int one(int w)
    {
        int sum = 0;
        for(int i = 0; i < w; i++)
        {
            sum += filter[i];
        }
        return sum;
    }
};
#endif