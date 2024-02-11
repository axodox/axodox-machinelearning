#pragma once
#include "StableDiffusionScheduler2.h"

namespace Axodox::MachineLearning
{
  class DpmPlusPlus2MScheduler : public StableDiffusionScheduler2
  {
  public:
    DpmPlusPlus2MScheduler(const StableDiffusionSchedulerOptions2& context);

    virtual Tensor ApplyStep(const Tensor& input, const Tensor& output, size_t step) override;

  private:
    Tensor _previousPredictedSample;
  };
}