#pragma once
#include "../../includes.h"

namespace Axodox::MachineLearning::Sessions
{
  class AXODOX_MACHINELEARNING_API OnnxModelSource
  {
  public:
    explicit OnnxModelSource(std::function<std::vector<uint8_t>()>&& source, const std::filesystem::path& pathHint = {});

    const std::filesystem::path& PathHint() const;

    std::vector<uint8_t> GetModelData() const;

    static std::unique_ptr<OnnxModelSource> FromFilePath(const std::filesystem::path& path);

#ifdef WINRT_Windows_Storage_H
    static std::unique_ptr<OnnxModelSource> FromStorageFile(const winrt::Windows::Storage::StorageFile& file);
#endif

  private:
    std::filesystem::path _pathHint;
    std::function<std::vector<uint8_t>()> _source;
  };
}