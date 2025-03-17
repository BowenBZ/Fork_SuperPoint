#include <opencv2/core.hpp>
#include <opencv2/flann/heap.h>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>

typedef struct {
    int x;
    int y;
    float confidence;
} CornerPoint;

typedef std::array<float, 256> Descriptor;

class SuperPointModel {
public: 
    SuperPointModel(std::string modelPath, float confidenceThresh) {
        _confidenceThresh = confidenceThresh;
        _initialized = false;
        try {
            _module = torch::jit::load(modelPath);
            _module.eval();
            _initialized = true;
        } catch (const c10::Error &e) {
            std::cerr << "Error loading the model\n";
        }
    }

    auto Initialized() const -> bool { return _initialized; }

    /*
    Input
      img - OpenCV Mat grayscale float32 input image in range [0,1].
    Output
      corners - vector of corners.
      desc - vector of unit unit normalized descriptors.
    */
    void Process(const cv::Mat& image, std::vector<CornerPoint>& corners, std::vector<Descriptor>& desc) {
        corners.clear();
        desc.clear();

        // 1. Process the input
        assert(image.channels() == 1);
        assert(image.type() == CV_32F);
        int imageHeight = image.rows;
        int imageWidth = image.cols;
        // Create a tensor from the cv::Mat. OpenCV Mat dimensions are [height, width, channels]
        torch::Tensor tensor_image = torch::from_blob(
            image.data,
            { imageHeight, imageWidth, image.channels() },
            torch::kFloat32
        );

        // Permute dimensions from HxWxC to CxHxW
        tensor_image = tensor_image.permute({2, 0, 1});

        // Add a batch dimension, so the input size is (1, 1, 240, 320)
        tensor_image = tensor_image.unsqueeze(0);

        // 2. Run the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
        auto outputs = _module.forward(inputs).toTuple();

        // Output size is (1, 65, 30, 40)
        torch::Tensor semi = outputs->elements()[0].toTensor();
        // Output size is (1, 256, 30, 40)
        torch::Tensor coarse_desc = outputs->elements()[1].toTensor();

        // 3. Process the output
        semi = semi.squeeze();

        // Softmax
        semi = torch::exp(semi);
        semi = semi / (semi.sum(0) + 0.00001);

        // Remove dustbin
        torch::Tensor nodust = semi.slice(0, 0, semi.size(0) - 1);
        nodust = nodust.permute({1, 2, 0});

        // Reshape to get full resolution heatmap (240, 320)
        int Hc = int(imageHeight / _cell);
        int Wc = int(imageWidth / _cell);
        torch::Tensor heatmap = torch::reshape(nodust, {Hc, Wc, _cell,_cell});
        heatmap = heatmap.permute({0, 2, 1, 3});
        heatmap = torch::reshape(heatmap, {Hc * _cell, Wc * _cell});

        // Select points with confidence higher than threshold
        auto selected = torch::where(heatmap >= _confidenceThresh);
        torch::Tensor ys = selected[0];
        torch::Tensor xs = selected[1];
        int selected_cnt = xs.size(0);
        // No points found
        if (selected_cnt == 0) {
            return;
        }

        // pts is (3, N)
        torch::Tensor pts = torch::zeros({3, selected_cnt});
        pts.index_put_({0}, xs);
        pts.index_put_({1}, ys);
        pts.index_put_({2}, heatmap.masked_select(heatmap >= _confidenceThresh));

        // TODO: # Apply NMS.
        // TODO: Sort by confidence.
        // TODO: Remove points along border.

        for (int i = 0; i < selected_cnt; i++) {
            corners.push_back({(int)xs[i].item<int64_t>(), (int)ys[i].item<int64_t>(), 1.f});
        }
    }
private:
    bool _initialized;
    torch::jit::script::Module _module;
    float _confidenceThresh;

    // Size of each output cell. Keep this fixed.
    int _cell = 8;
};

class VideoStreamer {
public:
    VideoStreamer(std::string datasetDir, int width, int height) {
        _size = cv::Size(width, height);
        _initialized = BuildFileIndex(datasetDir);
    }

    auto Initialized() const -> bool { return _initialized; }

    auto HasNextFrame() const -> bool { return _imageIdx < _rgbFilesPath.size(); }

    cv::Mat NextFrame() {
        if (!HasNextFrame()) {
            return cv::Mat();
        }

        cv::Mat image = cv::imread(_rgbFilesPath[_imageIdx++], cv::IMREAD_GRAYSCALE);
        cv::resize(image, image, _size, 0, 0, cv::INTER_LINEAR);

        // Input image is in CV_8U, need to convert to float type
        image.convertTo(image, CV_32F, 1.0 / 255);

        return image;
    }

private:
    bool _initialized;
    std::vector<std::string> _rgbFilesPath;
    size_t _imageIdx;
    cv::Size _size;

    bool BuildFileIndex(std::string datasetDir) {
        std::string datasetEntryFile = datasetDir + "/associate.txt";
        std::ifstream fin ( datasetEntryFile );
        if (!fin)
        {
            printf("please generate the associate file called associate.txt!\n");
            return false;
        }

        while (!fin.eof()) {
            std::string rgbTime, rgbFile, depthTime, depthFile;
            fin >> rgbTime >> rgbFile >> depthTime >> depthFile;
            if (rgbTime.size() == 0) {
                break;
            }

            _rgbFilesPath.push_back (datasetDir + "/" + rgbFile);

            if (!fin.good()) {
                break;
            }
        }

        fin.close();
        printf("Total %zu images from dataset\n", _rgbFilesPath.size());
        _imageIdx = 0;
        return true;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Need to provide model path & dataset directory path\n");
        return -1;
    }

    printf("Model path is %s\n", argv[1]);
    SuperPointModel model(argv[1], 0.015);
    if (!model.Initialized()) {
        printf("Failed to load pytorch model\n");
        return -1;
    }

    printf("Model loaded successfully\n");

    // Load dataset
    printf("Dataset directory path is %s\n", argv[2]);
    VideoStreamer videoStreamer(argv[2], 320, 240);
    if (!videoStreamer.Initialized()) {
        printf("Failed to initialize video streamer\n");
        return -1;
    }

    std::vector<CornerPoint> corners; 
    std::vector<Descriptor> desc;

    cv::Mat image = videoStreamer.NextFrame();
    model.Process(image, corners, desc);

    for (size_t i = 0; i < corners.size(); i++) {
        cv::Point2d pt(corners[i].x, corners[i].y);
        cv::circle(image, pt, 1, {0, 25, 0}, -1, cv::LINE_8);
    }

    // Display the image in a window named "Display Window".
    cv::imshow("Display Window", image);
    
    // Wait indefinitely for a key press.
    cv::waitKey(0);

    // Create a dummy input tensor (modify dimensions as required).
    // torch::Tensor input = torch::randn({1, 3, 224, 224});
    
    // Run the model (assuming the forward method expects one input).
    // at::Tensor output = module.forward({input}).toTensor();
    // std::cout << "Output: " << output << std::endl;

    return 0;
}

// #include <torch/torch.h>
// #include <iostream>

// int main() {
//   torch::Tensor tensor = torch::rand({2, 3});
//   std::cout << tensor << std::endl;
// }