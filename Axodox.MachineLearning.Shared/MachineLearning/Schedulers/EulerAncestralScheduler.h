#pragma once
#include "StableDiffusionScheduler2.h"

namespace Axodox::MachineLearning
{
  class EulerAncestralScheduler : public StableDiffusionScheduler2
  {
  public:
    EulerAncestralScheduler(const StableDiffusionSchedulerOptions2& context);

    virtual Tensor ApplyStep(const Tensor& input, const Tensor& output, size_t step) override;
  };
}