#pragma once
#include "StableDiffusionScheduler.h"

namespace Axodox::MachineLearning
{
  class DpmPlusPlus2MScheduler : public StableDiffusionScheduler
  {
  public:
    DpmPlusPlus2MScheduler(const StableDiffusionSchedulerOptions& context);

    virtual Tensor ApplyStep(const Tensor& input, const Tensor& output, size_t step) override;

  private:
    Tensor _previousPredictedSample;
  };
}