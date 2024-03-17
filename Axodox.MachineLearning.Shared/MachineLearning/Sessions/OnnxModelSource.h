#pragma once
#include "../../includes.h"

namespace Axodox::MachineLearning::Sessions
{
  class AXODOX_MACHINELEARNING_API OnnxModelSource
  {
  public:
    explicit OnnxModelSource(std::function<std::vector<uint8_t>()>&& source);

    std::vector<uint8_t> GetModelData() const;

    static OnnxModelSource FromFilePath(const std::filesystem::path& path);

#ifdef WINRT_Windows_Storage_H
    static OnnxModelSource FromStorageFile(const winrt::Windows::Storage::StorageFile& file);
#endif

  private:
    std::function<std::vector<uint8_t>()> _source;
  };
}