#ifndef DEEPFRAME_BLOB_H_
#define DEEPFRAME_BLOB_H_

#include <vector>
#include <string>
#include <allocator>
#include <boost/shared_ptr.hpp>


namespace DeepFrame {
  template<Dtype> class TMemory;


  template<class Dtype>
  class Blob {
  private:
    TMemory<Dtype> D;



  };

}



#endif
