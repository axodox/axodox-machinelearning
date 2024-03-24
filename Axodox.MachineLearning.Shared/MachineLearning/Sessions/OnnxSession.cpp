#include "pch.h"
#include "OnnxSession.h"
#include "MachineLearning/Executors/CpuExecutor.h"
#include "MachineLearning/Executors/DmlExecutor.h"

using namespace Axodox::Infrastructure;
using namespace Axodox::MachineLearning::Executors;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning::Sessions
{
  OnnxSessionContainer::OnnxSessionContainer(const OnnxSessionParameters& parameters) :
    _parameters(parameters),
    _session(nullptr)
  {
    if (!_parameters.IsValid()) throw logic_error("Some of the required session parameters have not been set.");

    //Subscribe to executor reset
    _deviceResetSubscription = _parameters.Executor->DeviceReset({ this, &OnnxSessionContainer::OnDeviceReset });

    //Initalize session
    EnsureSession();
  }

  OnnxSessionRef OnnxSessionContainer::Session()
  {
    return { _mutex, &_session };
  }

  OnnxEnvironment* OnnxSessionContainer::Environment() const
  {
    return _parameters.Environment.get();
  }

  void OnnxSessionContainer::EnsureSession()
  {
    //Esnure executor (might trigger reset)
    _parameters.Executor->Ensure();

    //If the session exists nothing to do
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

  void OnnxSessionContainer::OnDeviceReset(Executors::OnnxExecutor* executor)
  {
    unique_lock lock(_mutex);
    _session = Ort::Session{nullptr};

    EnsureSession();
  }

  OnnxSessionParameters OnnxSessionParameters::Create(const std::filesystem::path& path, OnnxExecutorType executorType)
  {
    unique_ptr<OnnxExecutor> executor;
    switch (executorType)
    {
    case OnnxExecutorType::Cpu:
      executor = make_unique<CpuExecutor>();
      break;
    case OnnxExecutorType::Dml:
      executor = make_unique<DmlExecutor>();
      break;
    }

    return OnnxSessionParameters{
      .Environment = make_shared<OnnxEnvironment>(),
      .Executor = move(executor),
      .ModelSource = OnnxModelSource::FromFilePath(path)
    };
  }

  bool OnnxSessionParameters::IsValid() const
  {
    return Environment && Executor && ModelSource;
  }

  OnnxSessionParameters::operator bool() const
  {
    return IsValid();
  }

  void OnnxSessionUnlock::operator()(Ort::Session* value)
  {
    value->Evict();
  }
}