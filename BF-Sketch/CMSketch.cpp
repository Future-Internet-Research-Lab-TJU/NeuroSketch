#include "CMSketch.hpp"
#include "datatypes.hpp"
#include "bloom_filter.hpp"
#include <string>
uint32_t hval1[4];

CMSketch::CMSketch(int depth, int width)
{
    cm_.depth = depth;
    cm_.width = width;
    cm_.buckets = new Bucket *[depth * width];
    
    for (int i = 0; i < depth * width; i++) 
    {
        cm_.buckets[i] = (Bucket*)calloc(1, sizeof(Bucket));
        memset(cm_.buckets[i], 0, sizeof(Bucket));
    }
}

CMSketch::~CMSketch()
{
    for (int i = 0; i < cm_.depth * cm_.width; i++)
        free(cm_.buckets[i]);
}

void CMSketch::Update_in(unsigned char* key, val_tp val, uint32_t* hval1)
{
    for (int i = 0; i < cm_.depth; i++)
    {
        int index = i * cm_.width + (hval1[i] % cm_.width);
        CMSketch::Bucket *buckets = cm_.buckets[index];
        buckets->flow_size += val;
    }
}

void CMSketch::Update_out(unsigned char* key, val_tp val, uint32_t* hval1)
{
    for (int i = 0; i < cm_.depth; i++)
    {
        int index = i * cm_.width + (hval1[i] % cm_.width);
        CMSketch::Bucket *buckets = cm_.buckets[index];
        buckets->flow_num += val;
        buckets->flow_size += val;
    }
}