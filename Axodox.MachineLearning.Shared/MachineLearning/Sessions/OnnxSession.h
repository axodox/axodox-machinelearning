#pragma once
#include "../../includes.h"
#include "../Executors/OnnxExecutor.h"
#include "OnnxEnvironment.h"
#include "OnnxModelSource.h"

namespace Axodox::MachineLearning::Sessions
{
  struct OnnxSessionParameters
  {
    std::shared_ptr<OnnxEnvironment> Environment;
    std::shared_ptr<Executors::OnnxExecutor> Executor;
    std::shared_ptr<OnnxModelSource> ModelSource;

    bool IsValid() const;
  };

  class OnnxSession
  {
  public:
    OnnxSession(const OnnxSessionParameters& parameters);

    Threading::locked_ptr<Ort::Session> TryLock();

  private:
    OnnxSessionParameters _parameters;
    Ort::Session _session;
    std::shared_mutex _mutex;
    Infrastructure::event_subscription _deviceResetSubscription;

    void EnsureSession();
    void OnDeviceReset(Executors::OnnxExecutor* executor);
  };
}