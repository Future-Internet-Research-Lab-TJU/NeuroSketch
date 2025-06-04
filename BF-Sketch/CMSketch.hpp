#ifndef CM
#define CM

#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "datatypes.hpp"
#include "hash.h"

class CMSketch
{

    typedef struct SBUCKET_type {

       uint16_t flow_num;
       uint32_t flow_size;

    } Bucket;
   
    struct CM_type
    {
        Bucket **buckets;

        int depth;
        int width;
    };
    

    public:
    CMSketch(int depth, int width);
    ~CMSketch();

    void Update_in(unsigned char* key, val_tp val, uint32_t* hval1);

    void Update_out(unsigned char* key, val_tp val, uint32_t* hval1);

    private:

    CM_type cm_;

};


#endif
