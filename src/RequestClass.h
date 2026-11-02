#ifndef PIMSIM_REQUEST_CLASS_H
#define PIMSIM_REQUEST_CLASS_H

namespace DRAMSim
{

// Shared by sparse kernels and future scheduling policies. The default scheduler deliberately
// ignores this classification, preserving existing behavior.
enum class RequestClass
{
    MATRIX_VALUE,
    MATRIX_INDEX,
    DENSE_VECTOR,
    RESULT,
    PIM_CONTROL,
    OTHER
};

}  // namespace DRAMSim

#endif
