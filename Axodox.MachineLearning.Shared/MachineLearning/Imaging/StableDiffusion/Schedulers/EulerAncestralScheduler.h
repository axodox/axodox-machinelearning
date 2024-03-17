#pragma once
#include "StableDiffusionScheduler.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion::Schedulers
{
  class EulerAncestralScheduler : public StableDiffusionScheduler
  {
  public:
    EulerAncestralScheduler(const StableDiffusionSchedulerOptions& context);

    virtual Tensor ApplyStep(const Tensor& input, const Tensor& output, size_t step) override;
  };
}