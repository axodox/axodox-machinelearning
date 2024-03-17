#include "pch.h"
#include "OnnxSession.h"

using namespace Axodox::Infrastructure;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Sessions
{
  OnnxSession::OnnxSession(const OnnxSessionParameters& parameters) :
    _parameters(parameters),
    _session(nullptr)
  {
    if (!_parameters.IsValid()) throw logic_error("Some of the required session parameters have not been set.");

    //Subscribe to executor reset
    _deviceResetSubscription = _parameters.Executor->DeviceReset({ this, &OnnxSession::OnDeviceReset });

    //Initalize session
    EnsureSession();
  }

  Threading::locked_ptr<Ort::Session> OnnxSession::TryLock()
  {
    return { _mutex, &_session };
  }

  void OnnxSession::EnsureSession()
  {
    //Esnure executor (might trigger reset)
    _parameters.Executor->Ensure();

    //If the session exists nothing to do
    unique_lock lock(_mutex);
    if (_session) return;

    //Get environment
    auto& environment = _parameters.Environment->Environment();

    //Define session options
    auto options = _parameters.Environment->DefaultSessionOptions();
    _parameters.Executor->Apply(options);

    //Read model
    auto buffer = _parameters.ModelSource->GetModelData();

    //Create session
    _session = { environment, buffer.data(), buffer.size(), options };
  }

  void OnnxSession::OnDeviceReset(Executors::OnnxExecutor* executor)
  {
    unique_lock lock(_mutex);
    _session = Session{nullptr};

    EnsureSession();
  }

  bool OnnxSessionParameters::IsValid() const
  {
    return Environment && Executor && ModelSource;
  }

}