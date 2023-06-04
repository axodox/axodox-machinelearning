#pragma once
#include "Json/JsonSerializer.h"

namespace Axodox::MachineLearning
{
	struct AXODOX_MACHINELEARNING_API SafetyCheckerCropSize : public Json::json_object_base
	{
		Json::json_property<uint32_t> Width;
		Json::json_property<uint32_t> Height;

		SafetyCheckerCropSize();
	};

	struct AXODOX_MACHINELEARNING_API SafetyCheckerSize : public Json::json_object_base
	{
		Json::json_property<uint32_t> ShortestEdge;

		SafetyCheckerSize();
	};

	struct AXODOX_MACHINELEARNING_API SafetyCheckerOptions : public Json::json_object_base
	{
		Json::json_property<SafetyCheckerCropSize> CropSize;
		Json::json_property<bool> DoCenterCrop;
		Json::json_property<bool> DoNormalize;
		Json::json_property<bool> DoRescale;
		Json::json_property<bool> DoResize;
		Json::json_property<std::vector<float>> ImageMean;
		Json::json_property<std::vector<float>> ImageStd;
		Json::json_property<int32_t> Resample;
		Json::json_property<float> RescaleFactor;
		Json::json_property<SafetyCheckerSize> Size;

		SafetyCheckerOptions();
	};
}