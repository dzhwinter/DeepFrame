# DeepFrame
===========
learning from outstanding frameworks

DeepFrame is an benchmark of open source deep learning framework. gathering some tips from their project and join in DeepFrame
These DeepFramework include:

TODO 

- Caffe: is famous deeplearning and have a lot users
- Caffe2:  yangqing improve his caffe.
- lasso and pylearn2:  python abstract. it is the most easy one to modify
- torch:  just want to review lua script. do not want to add torch nnnet into DeepFrame, just I like lua
- mxnnet: cxxnet and minerva combination, cxxnet implement based on mshadow abstraction, minerva give an example use DAG schedule, stream engine  need to explore

## Lasso and pylearn2
some tips from their jobs.

learning the mxnet design engine.
*  translate the *(a1, a2) into __func(op, arg1, arg2)__, then schedule DAG.
*  async operation abstract as three types:
   * wait
   * waitALL()
   * send()

