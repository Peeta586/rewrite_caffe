#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <algorithm>
#include <iterator>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"

#include "caffe/common.hpp"

namespace caffe {

    typedef boost::mt19937 rng_t;

    inline rng_t* caffe_rng() {
        // cast RNG--> caffe::rng_t 
        return static_cast<caffe::rng_t*>(Caffe::rng_stream().generator());
    }

    // Fisher-Yates algorithm
    // 用均匀分布交换迭代器之间的元素，每次均匀分布都是有大到小的区间选择的
    template <class RandomAccessIterator, class RandomGenerator>
    inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end, RandomGenerator* gen){
        // 获取RandomAccessIterator 迭代器的每一项的距离，或者最大容量
        typedef typename std::iterator_traits<RandomAccessIterator>::difference_type difference_type;
        // 均匀分布的实例化声明
        typedef typename boost::uniform_int<difference_type> dist_type;

        // 迭代器元素之间的距离类型是difference_type
        difference_type length = std::distance(begin, end);
        if (length <= 0) return;

        for(difference_type i = length - 1; i>0; --i){
            // dist_type这个是类，dist是该类的实例 
            // 指定0，i之间进行均匀分布
            // dist(*gen) 传入随机生成器，然后根据定义的0,i区间，产生随机的数
            dist_type dist(0,i);
            // 然后交换iter之间的值
            std::iter_swap(begin + i, begin + dist(*gen));
        }
    }

    template <class RandomAccessIterator>
    inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
        shuffle(begin, end, caffe_rng());
    }

} // namespace caffe


#endif // CAFFE_RNG_CPP_HPP_