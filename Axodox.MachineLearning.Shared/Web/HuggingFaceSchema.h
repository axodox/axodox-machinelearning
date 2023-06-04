#pragma once
#ifdef PLATFORM_WINDOWS
#include "Json/JsonSerializer.h"

namespace Axodox::Web
{
  struct AXODOX_MACHINELEARNING_API HuggingFaceModelInfo : public Json::json_object_base
  {
    Json::json_property<std::string> Id;

    HuggingFaceModelInfo();
  };

  struct AXODOX_MACHINELEARNING_API HuggingFaceFileRef : public Json::json_object_base
  {
    Json::json_property<std::string> FilePath;

    HuggingFaceFileRef();
  };

  struct AXODOX_MACHINELEARNING_API HuggingFaceModelDetails : public Json::json_object_base
  {
    static const std::set<std::string> StableDiffusionOnnxFileset;

    Json::json_property<std::string> Id;
    Json::json_property<std::string> Author;
    Json::json_property<uint32_t> Downloads;
    Json::json_property<uint32_t> Likes;
    Json::json_property<std::vector<std::string>> Tags;
    Json::json_property<std::vector<HuggingFaceFileRef>> Files;

    HuggingFaceModelDetails();

    bool IsValidModel(const std::set<std::string>& fileset);
  };
}
#endif