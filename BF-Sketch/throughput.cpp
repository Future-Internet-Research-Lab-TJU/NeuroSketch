#include <unordered_map>
#include <utility>
#include <iomanip>
#include <cmath>
#include <fstream>
#include "CMSketch.hpp"
#include "bloom_filter.hpp"
#include "adaptor.hpp"
#include "datatypes.hpp"
#include "util.h"

#define tot_mem 500
#define bf_mem 50
#define cm_depth 3 //number of rows

int main(int argc, char* argv[]) {
    //Confiture parameter

    //Dataset filename
    const char* filenames = "iptraces.txt";
    unsigned long long buf_size = 5000000000;


    // sketch parameters
    const int width = bf_mem * 8 * 1024; //number of bits in bloom filter
    int cm_width = (tot_mem - bf_mem) * 1024 / 5 / 3; //number of buckets in each row
    
    //evaluation
    std::ifstream tracefiles(filenames);
    if (!tracefiles.is_open()) {
        std::cout << "Error opening file" << std::endl;
        return -1;
    }

    for (std::string file; getline(tracefiles, file);) 
    {
        //load traces
        double AE = 0;
        double RE = 0;
        uint32_t hval1[4];      
        Adaptor* adaptor =  new Adaptor(file, buf_size);
        std::cout << "[Dataset]: " << file << std::endl;
        std::cout << "[Message] Finish read data." << std::endl;

        //Get ground
        adaptor->Reset();
        mymap ground;
        val_tp sum = 0;
        tuple_t t;
        while(adaptor->GetNext(&t) == 1) //store flow key-flow size pair
        { 
            sum += 1;
            key_tp key;
            memcpy(key.key, &(t.key), LGN);
            if (ground.find(key) != ground.end()) {
                ground[key] += 1;
            } 
            else {
                ground[key] = 1;
            }
        }

        std::cout << "[Message] NO.Flows: " << ground.size() << " NO.Packets: " << sum << std::endl;

        //Create structure
         BloomFilter<width>* bf = NULL;
        CMSketch* cm = new CMSketch(cm_depth, cm_width);
        bf = new BloomFilter<width>();

        //Update sketch
        adaptor->Reset();
        memset(&t, 0, sizeof(tuple_t));
        bool contains;
        uint64_t t1 = 0, t2 = 0;
      
        double throughput = 0;
        t1 = now_us();
        while(adaptor->GetNext(&t) == 1) 
        {
            contains = bf->contains((unsigned char*)&(t.key), LGN, hval1);
            if (contains) 
            {
                cm->Update_in((unsigned char*)&(t.key), 1, hval1); 
            }
            else
            {
                cm->Update_out((unsigned char*)&(t.key), 1, hval1);
            }
        }
        t2 = now_us();
        throughput = adaptor->GetDataSize() / (double)(t2 - t1) * 1000000000 / 1000000 ;

        delete cm;
        delete bf;
        delete adaptor;
        std::cout  << throughput << std::endl;
    }
}
