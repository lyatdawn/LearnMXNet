/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cc
 * \brief
 * \author Bing Xu
*/

#include "./batch_norm-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_batch_norm-inl.h"
#endif  // MXNET_USE_MKL2017 // 是否使用MKL2017, 默认在make mxne的时候不使用. 在src/operator/mkl对
// BN, concat, convolution, FC, lrn, pooling, relu等进行了重新的定义. 这些定义是基于MKL2017的. 

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(BatchNormParam param, int dtype) {// 创建op, 传入参数param和dtype. 是dropo-inl.h中类:
  // Operator *CreateOp(DropoutParam param, int dtype);的实现.  
#if MXNET_USE_MKL2017 == 1
  if (!param.use_global_stats) {
    return new MKLBatchNormOp<cpu, float>(param);
  } else {
    if (enableMKLWarnGenerated())
      LOG(INFO) << MKLBatchNormOp<cpu, float>::getName() << " Skip MKL optimization";
  }
#endif // 使用MKL2017库. 
  return new BatchNormOp<cpu>(param); // 不使用MKL2017库时的创建OP. 
  // 调用BatchNormOp类的构造函数explicit BatchNormOp(BatchNormParam param) {} 来创建类BatchNormOp的对象op. 
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BatchNormProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

/*
使用下面的宏定义来将parameter结构和OperatorProperty类注册到MXNet系统中:
DMLC_REGISTER_PARAMETER和MXNET_REGISTER_OP_PROPERTY. 
*/
DMLC_REGISTER_PARAMETER(BatchNormParam);

MXNET_REGISTER_OP_PROPERTY(BatchNorm, BatchNormProp)
.describe("Apply batch normalization to input.") // 调用describe函数描述. 
.add_argument("data", "Symbol", "Input data to batch normalization") // 调用add_argument, 给BN层添加参数.  
.add_argument("gamma", "Symbol", "gamma matrix")
.add_argument("beta", "Symbol", "beta matrix") // 参数有data, gamma, beta. 
.add_arguments(BatchNormParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

