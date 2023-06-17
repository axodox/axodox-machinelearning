#pragma once
#ifdef PLATFORM_WINDOWS
#include "../includes.h"
#include "HuggingFaceSchema.h"
#include "Threading/AsyncOperation.h"

namespace Axodox::Web
{
  class AXODOX_MACHINELEARNING_API HuggingFaceClient
  {
  public:
    HuggingFaceClient();

    std::vector<std::string> GetModels(std::string_view filter);

    std::optional<HuggingFaceModelDetails> GetModel(std::string_view modelId);

    bool TryDownloadModel(std::string_view modelId, const std::set<std::string>& fileset, const std::filesystem::path& targetPath, Threading::async_operation& operation);

  private:
    static const wchar_t* const _baseUri;
    winrt::Windows::Web::Http::HttpClient _httpClient;

    static winrt::Windows::Web::Http::HttpClient CreateClient();

    std::string TryQuery(std::string_view uri);
  };
}
#endif