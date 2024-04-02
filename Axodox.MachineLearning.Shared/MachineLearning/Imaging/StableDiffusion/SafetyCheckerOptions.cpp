#include "pch.h"
#include "SafetyCheckerOptions.h"

namespace Axodox::MachineLearning::Imaging::StableDiffusion
{
  SafetyCheckerCropSize::SafetyCheckerCropSize() :
    Width(this, "width", 224),
    Height(this, "height", 224)
  { }

  SafetyCheckerSize::SafetyCheckerSize() :
    ShortestEdge(this, "shortest_edge", 224)
  { }

  SafetyCheckerOptions::SafetyCheckerOptions() :
    CropSize(this, "crop_size"),
    DoCenterCrop(this, "do_center_crop", true),
    DoNormalize(this, "do_normalize", true),
    DoRescale(this, "do_rescale", true),
    DoResize(this, "do_resize", true),
    ImageMean(this, "image_mean", { 0.48145466f, 0.4578275f, 0.40821073f }),
    ImageStd(this, "image_std", { 0.26862954f, 0.26130258f, 0.27577711f }),
    Resample(this, "resample"),
    RescaleFactor(this, "rescale_factor", 1.f / 255.f),
    Size(this, "size")
  { }
}