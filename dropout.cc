/*!
 * Copyright (c) 2015 by Contributors
 * \file dropout.cc
 * \brief
 * \author Bing Xu
*/

#include "./dropout-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(DropoutParam param, int dtype) { // 创建op, 传入参数param和dtype. 是batcn_norm-inl.h中类:
  // Operator *CreateOp(BatchNormParam param, int dtype);的实现.  
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DropoutOp<cpu, DType>(param); // 调用DropoutOp类的构造函数explicit DropoutOp(DropoutParam param)类创建类的对象.  
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *DropoutProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                              std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

/*
使用下面的宏定义来将parameter结构和OperatorProperty类注册到MXNet系统中:
DMLC_REGISTER_PARAMETER和MXNET_REGISTER_OP_PROPERTY. 
*/
DMLC_REGISTER_PARAMETER(DropoutParam);

MXNET_REGISTER_OP_PROPERTY(Dropout, DropoutProp)
.describe(R"(Apply dropout to input. 
During training, each element of the input is randomly set to zero with probability p.
And then the whole tensor is rescaled by 1/(1-p) to keep the expectation the same as
before applying dropout. During the test time, this behaves as an identity map.
)") // 将Dropout应用到input上, 即Dropout是对于输入的结点数据来说的. 
.add_argument("data", "Symbol", "Input data to dropout.") // Dropout层的参数. 
.add_arguments(DropoutParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


