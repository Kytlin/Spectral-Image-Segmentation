// Smoke test: the data folder holds at least one image to work with.

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace fs = std::filesystem;

TEST(SampleData, DataFolderContainsAnImage) {
    const fs::path data_dir(SEG_DATA_DIR);
    ASSERT_TRUE(fs::is_directory(data_dir)) << "missing " << data_dir;

    int images = 0;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (entry.is_regular_file() && entry.file_size() > 0 &&
            (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")) {
            ++images;
        }
    }
    EXPECT_GT(images, 0) << "no .png/.jpg/.jpeg/.bmp file in " << data_dir;
}
