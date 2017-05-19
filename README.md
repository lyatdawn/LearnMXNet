   MXNet 中的operator既包含了实际的计算也包含了一些附加信息的 class, 如Layer的参数(Param), Layer的属性(Prop)等, 这些附加信息可以帮助我们的系统
来实现原地更新和自动微分等优化. mxnet所有operator的计算都是基于系统提供的数据结构 mshadow::TBob, 输入是TBlob的数据, 输出是Tensor的数据. TBlob这
一数据结构类似于张量(Tensor), 只是TBlob更灵活. 现对mxnet的operator的学习记录进行详细说明. 首先对mxnet的Activation操作进行详细说明, 源码见
src/operator/activation-inl.h. 为了说明激活函数的操作情况, 本文将softplus激活函数也做成了mxnet的Layer(mxnet本身已经实现了softplus激活函数的功
能, 明后才能是SoftRelu). 现将源码softplus-inl.h及softplus.cc; batch_norm-inl.h及batch_norm.cc; fully_connected-inl.h及
fully_connected.cc; convolution-inl.h及convolution.cc; pooling-inl.h及pooling.cc; dropout-inl.h及dropout.cc贴上. 源码的注释都是笔者自己
写的, 有分析不对的地方网各位读者加以指正. 
