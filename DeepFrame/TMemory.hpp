#ifndef DEEPFRAME_TMEMORY_H_
#define DEEPFRAME_TMEMORY_H_


namespace DeepFrame {
  class TMemory {
  public:

    const void* cpu_data();
    void *set_cpu_data(void* data);
    const void* gpu_data();
    void *set_gpu_data(void* data);

    void* mutable_cpu_data();
    void* mutable_gpu_data();
    enum SynceHead {lUNINTIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED};
    SynceHead head() {return head_;}

  private:
    void to_cpu();
    void to_gpu();
    void* cpu_ptr_;
    void* gpu_ptr_;
    SynceHead head_;
    bool own_cpu_data;
    bool own_gpu_data;
    int gpu_device;


  };
}
template<class Dtype>
class TMemory {
private:

}

#endif
