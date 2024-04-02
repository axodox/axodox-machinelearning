#include "pch.h"
#ifdef PLATFORM_WINDOWS
#include "HuggingFaceClient.h"
#include "Infrastructure/Win32.h"
#include "Threading/AsyncOperation.h"

using namespace Axodox::Infrastructure;
using namespace Axodox::Json;
using namespace Axodox::Threading;
using namespace std;
using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Security::Cryptography;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Web::Http;
using namespace winrt::Windows::Web::Http::Headers;
using namespace winrt::Windows::Web::Http::Filters;

namespace Axodox::MachineLearning::Web
{
  const wchar_t* const HuggingFaceClient::_baseUri = L"https://huggingface.co/";

  HuggingFaceClient::HuggingFaceClient() :
    _httpClient(CreateClient())
  { }

  std::vector<std::string> HuggingFaceClient::GetModels(std::string_view filter)
  {
    auto result = TryQuery("models?filter=" + string(filter));
    auto models = try_parse_json<vector<HuggingFaceModelInfo>>(result);
    if (!models) return {};

    vector<string> results;
    results.reserve(models->size());
    for (auto& model : *models)
    {
      results.push_back(*model.Id);
    }

    return results;
  }

  std::optional<HuggingFaceModelDetails> HuggingFaceClient::GetModel(std::string_view modelId)
  {
    auto result = TryQuery("models/" + string(modelId));
    return try_parse_json<HuggingFaceModelDetails>(result);
  }

  bool HuggingFaceClient::TryDownloadModel(std::string_view modelId, const std::set<std::string>& fileset, const std::set<std::string>& optionals, const std::filesystem::path& targetPath, Threading::async_operation& operation)
  {
    async_operation_source async;
    operation.set_source(async);

    async.update_state("Fetching model metadata...");
    auto modelInfo = GetModel(modelId);
    if (!modelInfo)
    {
      async.update_state(async_operation_state::failed, "Failed to fetch model metadata.");
      return false;
    }

    if (!modelInfo->IsValidModel(fileset))
    {
      async.update_state(async_operation_state::failed, "Model is missing required files.");
      return false;
    }

    async.update_state("Ensuring output directory...");
    error_code ec;
    if (!filesystem::exists(targetPath, ec))
    {
      filesystem::create_directories(targetPath, ec);
      if (ec)
      {
        async.update_state(async_operation_state::failed, "Failed to create output directory.");
        return false;
      }
    }

    try
    {
      auto files = fileset;

      for (auto& file : *modelInfo->Files)
      {
        if (optionals.contains(*file.FilePath)) files.emplace(*file.FilePath);
      }

      auto fileCount = files.size();
      size_t fileIndex = 0;
      for (auto& file : files)
      {
        //Exit loop if cancelled
        if (async.is_cancelled()) break;

        //Update state
        async.update_state(float(fileIndex) / (fileCount - 1), format("Downloading {} ({}/{})...", file, fileIndex + 1, fileCount));
        
        //Execute
        auto requestResult = _httpClient.TryGetInputStreamAsync(Uri{ _baseUri + to_wstring(format("{}/resolve/main/{}", modelId, file)) }).get();
        if (!requestResult.Succeeded())
        {
          async.update_state(async_operation_state::failed, format("Failed to download {}.", file));
          return false;
        }
        
        //Parse response
        auto response = requestResult.ResponseMessage();
        if (response.StatusCode() != HttpStatusCode::Ok)
        {
          async.update_state(async_operation_state::failed, format("Failed to download {}.", file));
          return false;
        }

        //Ensure folder
        auto targetFilePath = (targetPath / file).make_preferred();
        auto targetFolderPath = targetFilePath.parent_path();
        if (!filesystem::exists(targetFolderPath, ec))
        {
          filesystem::create_directory(targetFolderPath, ec);
          if (ec)
          {
            async.update_state(async_operation_state::failed, "Failed to create output directory.");
            return false;
          }
        }

        //Create file
        auto targetFolder = StorageFolder::GetFolderFromPathAsync(targetFolderPath.c_str()).get();

        auto targetFile = targetFolder.CreateFileAsync(targetFilePath.filename().c_str(), CreationCollisionOption::ReplaceExisting).get();
        auto targetStream = targetFile.OpenAsync(FileAccessMode::ReadWrite).get();

        //Copy to disk        
        auto content = response.Content();
        auto lengthHeader = content.Headers().TryLookup(L"Content-Length");
        uint64_t length = lengthHeader ? stoll(to_string(lengthHeader.value())) : 0ul;
        
        uint64_t position = 0;
        auto sourceStream = content.ReadAsInputStreamAsync().get();
        
        Buffer buffer{ 1024 * 1024 };
        while (!async.is_cancelled())
        {
          auto bufferRead = sourceStream.ReadAsync(buffer, buffer.Capacity(), InputStreamOptions::None).get();
          
          position += bufferRead.Length();
          async.update_state(format("Downloading {} ({}/{} MB)...", file, position / 1024 / 1024, length / 1024 / 1024));
          
          if (bufferRead.Length() == 0) break;

          targetStream.WriteAsync(bufferRead).get();
        }

        sourceStream.Close();
        targetStream.Close();
        fileIndex++;
      }

      if (!async.is_cancelled())
      {
        async.update_state(async_operation_state::succeeded, 1.f, "Model downloaded successfully.");
      }
      else
      {
        async.update_state(async_operation_state::cancelled, 1.f, "Operation cancelled.");

        for (auto& file : files)
        {
          auto targetFilePath = (targetPath / file).make_preferred();

          error_code ec;
          filesystem::remove(targetFilePath, ec);
        }
      }

      return !async.is_cancelled();
    }
    catch (const hresult_error& error)
    {
      async.update_state(async_operation_state::failed, to_string(error.message()));
      return false;
    }
    catch (...)
    {
      async.update_state(async_operation_state::failed, "Unknown error.");
      return false;
    }
  }

  winrt::Windows::Web::Http::HttpClient HuggingFaceClient::CreateClient()
  {
    HttpBaseProtocolFilter filter{};
    filter.AllowUI(false);

    auto cacheControl = filter.CacheControl();
    cacheControl.ReadBehavior(HttpCacheReadBehavior::MostRecent);
    cacheControl.WriteBehavior(HttpCacheWriteBehavior::NoCache);
    
    return HttpClient{ filter };
  }

  std::string HuggingFaceClient::TryQuery(std::string_view uri)
  {
    string result;

    try
    {
      //Create request
      HttpRequestMessage request{};
      {
        request.RequestUri(Uri{ _baseUri + to_wstring(format("api/{}", uri)) });
        request.Method(HttpMethod::Get());
        
        const auto& headers = request.Headers();
        headers.Accept().TryParseAdd(L"application/json");
      }

      //Execute
      auto requestResult = _httpClient.TrySendRequestAsync(request).get();
      if (!requestResult.Succeeded()) return "";

      //Parse response
      auto response = requestResult.ResponseMessage();
      if (response.StatusCode() != HttpStatusCode::Ok) return "";

      result = to_string(response.Content().ReadAsStringAsync().get());
    }
    catch (...)
    {
      return "";
    }

    return result;
  }
}
#endif