/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"

#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_fully_connected-inl.h"
#endif  // MXNET_USE_MKL2017 是否使用MKL2017, 在src/operator/mkl文件夹下定义了MKL下的一些常见操作:
/*
BN, concat, convolution, FC, LRN, pooling, relu等. 这和src/operator/下定义的BN, concat, convolution, FC, LRN, pooling, relu等
这些操作虽是相同, 但是定义是有区别的. 
*/ 

#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_fully_connected-inl.h"
#endif  // MXNET_USE_NNPACK
/*
是否使用nnpack, NNPACK由facebook开发, 是一个加速神经网络计算的加速包, NNPACK可以在多核CPU平台上提高卷积层计算性能. 在
src/operator/nnpack下定义了convolution, FC, LRN, pooling操作. 
*/

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FullyConnectedParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  switch (dtype) {
  case mshadow::kFloat32:
    return new MKLFullyConnectedOp<cpu, float>(param);
  case mshadow::kFloat64:
    return new MKLFullyConnectedOp<cpu, double>(param);
  default:
    LOG(INFO) << MKLFullyConnectedOp<cpu, float>::getName() << " Skip MKL optimization";
    break;
  }
#endif
// 是否使用MKL2017, 利用MKL下的FC操作类定义OP操作. dtype是int型的变量, 取值有mshadow::kFloat32, mshadow::kFloat64还有其他, 
// 不同的dtype创建不同类型(float或double)型的FC操作. 
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  // nnp_fully_connected_inference will do optimization for batch-size = 1
  // nnp_fully_connected_output will do optimization for batch-size > 1
  // but just found FullyConnected in NNPACK result is wrong when batch_size != 2^n
  // so here only using NNPACK when batch_size = 2^n.
  if ((batch_size == 1) || ((batch_size > 1) && (!(batch_size & (batch_size - 1))))) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKFullyConnectedOp<cpu, float>(param);
    default:
      break;
    }
  }
#endif
// 是否使用nnpack创建FC操作.

// 不使用MKL和NNPACK下创建FC操作. 在创建FC操作类时, 传入参数param.  
  switch (dtype) {
  case mshadow::kFloat32:
    op = new FullyConnectedOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new FullyConnectedOp<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully connected layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  // 创建32位float型的CPU FC操作或64位float(double)的CPU FC操作. 

  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *FullyConnectedProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape(1, TShape()), aux_shape;
  std::vector<int> out_type(1, -1), aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(FullyConnected, FullyConnectedProp)
.describe(R"(Apply matrix multiplication to input then add a bias.
It maps the input of shape `(batch_size, input_dim)` to the shape of
`(batch_size, num_hidden)`. Learnable parameters include the weights
of the linear transform and an optional bias vector.)")
.add_argument("data", "Symbol", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(FullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
