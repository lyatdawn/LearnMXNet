/*!
 * Copyright (c) 2017 by Contributors
 * \file softplus.cc
 * \brief softplus op
 * \author L
*/
#include "./softplus-inl.h"
#include "./mshadow_op.h" 
/*
mshadow_op.h文件. mshadow_op是一个命名空间, 主要声明了一些结构体, 例如sigmoid, sigmoid_grad, relu, relu_grad. 
这些激活函数的结构体是用来实现
激活函数的功能的, 如relu激活函数:
DType(a > DType(0.0f) ? a : DType(0.0f)) // 利用Dtype做强制类型转换, 那么relu函数的功能就是max(x, 0).
在调用这些结构体的时, 是通过:
op = new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
做的, 即新建一个操作op, 指定device, ForwardOp, BackwardOp. ForwardOp, BackwardOp就是relu, relu_grad, 这样也就指定了激活
函数的前向和后向的具体操作了. 即指定新建的op的功能.     
*/
#if MXNET_USE_MKL2017 == 1 //  Intel数学核心函数库(MKL). mxnet中关于MKL库的文件均在src/operator/mkl下,  
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
// #include "./mkl/mkl_relu-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftplusParam param, int dtype) { // 创建op, 传入参数param和dtype, dtype可以是 mshadow::kFloat32, 0;
// mshadow::kFloat64, 为1. 
  Operator *op = NULL; // 声明新的op为空, 然后通过-inl.h中定义好的类SoftplusOp来定义op. 
#if MXNET_USE_MKL2017 == 1
  /*if (param.act_type == activation::kReLU) {
      switch (dtype) {
      case mshadow::kFloat32:
          return new MKLReluOp<cpu, float>();
      case mshadow::kFloat64:
          return new MKLReluOp<cpu, double>();
      default:
          break;
      }
  }// 由于激活函数是Softplus, 因此param.act_type == softplus. 因此, 这可以注释了. 
  */
  if (enableMKLWarnGenerated())
    LOG(INFO) << MKLReluOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { // MSHADOW_REAL_TYPE_SWITCH宏.  
    switch (param.act_type) {
      case softplus::kSoftplus:
        op = new SoftplusOp<cpu, mshadow_op::softplus, mshadow_op::softplus_grad, DType>();
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  })
  return op; 
  /*
  op即是定义的新layer, 即operator. 利用-inl.h中定义好的SoftplusOp类来实例化op, SoftplusOp类已定义好了前向和反向操作, 因此op就
  可以进行前向和反向操作了. 
  
  param.act_type. SoftplusParam的对象访问SoftplusParam成员act_type, 用来获得激活函数的类型. 
  已在-inl.h中, 声明act_type时, 利用add_num函数为act_type添加了softplus.  
  
  op = new SoftplusOp<cpu, mshadow_op::softplus, mshadow_op::softplus_grad, DType>();
  新建一个操作op, 指定device, ForwardOp, BackwardOp. ForwardOp, BackwardOp就是softplus, softplus_grad, 这样也就指定了激活
  函数的前向和后向的具体操作了. 即指定新建的op的功能.     
  
  mshadow_op::softplus, mshadow_op::softplus_grad是在./mshadow_op.h定义的结构体, 里面定义好了前向和反向操作. -inl.h中的
  ForwardOp即mshadow_op::softplus, BackwardOp即mshadow_op::softplus_grad. 
  */
   
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftplusProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  
  /*
  out_shape和out_type是在.cc中定义的, 然后调用-inl.h中的InferType和InferShape来指定out_type和out_shape. 
  */
  CHECK(InferType(in_type, &out_type, &aux_type)); // 检查InferType, 即检查in_type, out_type, aux_type是否推断正确. 
  CHECK(InferShape(in_shape, &out_shape, &aux_shape)); // 检查InferShape, 即检查in_shape, out_shape, aux_shape是否推断正确.
  // 推断正确, InferType和InferShape返回True, 否则返回False. 
   
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}
/*
-inl.h中只是声明了Operator *XXProp::CreateOperatorEx函数, 其具体实现在.cc中.

DO_BIND_DISPATCH comes from operator_common.h, 声明为了宏. 
#if MXNET_USE_CUDA // GPU
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
      return Method<cpu>(__VA_ARGS__);                               \
    } else {                                                         \
      return Method<gpu>(__VA_ARGS__);                               \
    }
#else // CPU
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
    return Method<cpu>(__VA_ARGS__);                                 \
  } else {                                                           \
    LOG(FATAL) << "GPU is not enabled";                              \
    return nullptr;                                                  \
  }
#endif
*/


/*
使用下面的宏定义来将parameter结构和OperatorProperty类注册到MXNet系统中:
DMLC_REGISTER_PARAMETER和MXNET_REGISTER_OP_PROPERTY. 
*/
DMLC_REGISTER_PARAMETER(SoftplusParam);

MXNET_REGISTER_OP_PROPERTY(Softplus, SoftplusProp)
.describe(R"(Elementwise activation function.
// describe描述函数, 是对该操作的描述, 字符串. 
The following activation types is supported (operations are applied elementwisely to each
scalar of the input tensor):

- `softplus`: SoftPlus, `y = log(1 + exp(x))`

See `LeakyReLU` for other activations with parameters.
)") 
.add_argument("data", "Symbol", "Input data to activation function.") // add_argument添加参数.  
.add_arguments(SoftplusParam::__FIELDS__()); // __FIELDS__:
/*
SoftplusParam继承的是Parameter, Parameter结构体定义的__FIELDS__函数:
inline static std::vector<ParamFieldInfo> __FIELDS__() {...}
功能: get the fields of the parameters.
*/

 
/*
在" ... "内部的均是字符串, 即是新的OP的帮助信息, help(mxnet.sym.Softplus)时则会输出这些信息. 不支持中文. 
*/

}  // namespace op
}  // namespace mxnet
