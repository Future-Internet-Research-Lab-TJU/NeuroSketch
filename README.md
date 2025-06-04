# NeuroSketch: Bloom filter-based Sketch for Accurate Network Measurement via Neural Networks

---
### Paper

__Jindian Liu, Zhuo Li, Hao Xun, Yu Zhang, Peng Luo, Qiang Li, and Kaihua Liu.__ 
NeuroSketch: Bloom filter-based Sketch for Accurate Network Measurement via Neural Networks.
_IEEE-ACM Transactions on Networking_

---
### Files
- Top.h: the top part of funnel sketch
- Middle.h: the middle part of funnel sketch
- Bottom.h：the bottom part of funnel sketch
- Funnelsketch.h：the implementation of funnelsketch
- param.h: the parameters of funnel sketch
- BOBHash32.h: the hash functions
- frequency.cpp：flow size estiamtion
- heavy_changer.cpp: heavy changer detection
- heavy_hitter.cpp: heavy hitter detection
- throughput.cpp: throughput
---

### Compile and Run the examples
Funnel Sketch is implemented with C++. We show how to compile the examples on
Ubuntu with g++ and make.

#### Requirements
- Ensure __g++__ and __make__ are installed.  Our experimental platform is
  equipped with Ubuntu 20.04

  - Prepare the test files.
   - You can use Real-life Transactional, caida, Zipf distribution and so on to test its performance.


   #### Compile
- Compile examples with make

```
    $ make frequency
    $ make heavy_changer
    $ make heavy_hitter
    $ make throughput
```

- The g++ compiler should be suitable 


#### Run
- Run the examples, and the program will output some statistics. 

```
    $ ./frequency
    $ ./heavy_change
    $ ./heavy_hitter
    $ ./throughput
```
