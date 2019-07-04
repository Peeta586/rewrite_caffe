#ifndef INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_
#define INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_

#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

/*
可能处理proto信号是需要一些处理，检查Solver的配置
 */
namespace caffe {

class SignalHandler {
    public:
    // constructor. specify what action to take when a signal is received.
    SignalHandler(SolverAction::Enum SIGINT_action, 
                  SolverAction::Enum SIGHUP_action);
    ~SignalHandler();
    ActionCallback GetActionFunction();
    private:

    SolverAction::Enum CheckForSignals() const;
    SolverAction::Enum SIGINT_action_;
    SolverAction::Enum SIGHUP_action_;
}; //class SignalHandler

} // namespace caffe

#endif  // INCLUDE_CAFFE_UTIL_SIGNAL_HANDLER_H_