#include "superpoint_model.hpp"

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include <string>
#include <iostream>

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

/*
Input:
prev_desc (256, N1)
curr_desc (256, N2)

Output:
matched/unmatched index of curr_desc
*/
void find_matched_points(const cv::Mat& prev_desc,
                        const cv::Mat& curr_desc,
                        const float nn_thresh,
                        std::vector<int>& matchedIndices,
                        std::vector<int>& unmatchedIndices) {
    matchedIndices = cv::Mat();
    unmatchedIndices = cv::Mat();

    // If no previous descriptors, return empty matched points and all current points as unmatched.
    if (prev_desc.empty()) {
        for (int j = 0; j < curr_desc.cols; j++) {
            unmatchedIndices.push_back(j);
        }
        return;
    }

    // Dimensions:
    int N1 = prev_desc.cols;
    int N2 = curr_desc.cols;

    cv::Mat prev_desc_T;
    cv::transpose(prev_desc, prev_desc_T);
    // (N1, N2)
    cv::Mat dot_product = prev_desc_T * curr_desc;

    // Compute L2 distance = sqrt(2 - 2 * dot_product) element-wise.
    // Use element-wise operations.
    cv::Mat distance = 2 - 2 * dot_product;
    cv::sqrt(distance, distance);

    // For each previous descriptor (row of 'distance'), find the current descriptor index with the minimum distance.
    std::vector<int> matched_curr_indices(N1, -1);
    for (int i = 0; i < N1; i++) {
        float min_val = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j = 0; j < N2; j++) {
            float val = distance.at<float>(i, j);
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        matched_curr_indices[i] = min_idx;
    }

    // For each current descriptor (column of 'distance'), find the previous descriptor index with the minimum distance.
    std::vector<int> matched_prev_indices(N2, -1);
    for (int j = 0; j < N2; j++) {
        float min_val = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int i = 0; i < N1; i++) {
            float val = distance.at<float>(i, j);
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        matched_prev_indices[j] = min_idx;
    }

    // Create a vector of current point indices: 0, 1, ..., N2-1.
    // Compute bidirectional matches: for each current descriptor j, check if:
    //   j == matched_curr_indices[ matched_prev_indices[j] ]
    std::vector<bool> bidirectional_match(N2, false);
    for (int j = 0; j < N2; j++) {
        int prev_idx = matched_prev_indices[j];
        if (prev_idx >= 0 && prev_idx < N1) {
            bidirectional_match[j] = (j == matched_curr_indices[prev_idx]);
        }
    }

    // For each current descriptor j, check if its matching score is below the threshold.
    std::vector<bool> pass_thresh_match(N2, false);
    for (int j = 0; j < N2; j++) {
        int prev_idx = matched_prev_indices[j];
        float score = distance.at<float>(prev_idx, j);
        pass_thresh_match[j] = (score < nn_thresh);
    }

    // Final match: for each current point j, it is a valid match if both conditions are true.
    for (int j = 0; j < N2; j++) {
        if (bidirectional_match[j] && pass_thresh_match[j]) {
            matchedIndices.push_back(j);
        } else {
            unmatchedIndices.push_back(j);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Need to provide model path & dataset directory path\n");
        return -1;
    }

    printf("Model path is %s\n", argv[1]);
    SuperPointModel model(argv[1], 0.015, 4);
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

    std::vector<CornerPoint> prevCorners; 
    cv::Mat prevDesc;
    std::vector<int> matchedIndices;
    std::vector<int> unmatchedIndices;

    while (videoStreamer.HasNextFrame()) {
        cv::Mat image = videoStreamer.NextFrame();

        std::vector<CornerPoint> currCorners; 
        cv::Mat currDesc;
        model.Process(image, currCorners, currDesc);

        // Find matches 
        find_matched_points(prevDesc, currDesc, 0.7, matchedIndices, unmatchedIndices);

        // Convert image back to RGB for display
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

        for (const auto& idx: matchedIndices) {
            cv::Point2d pt(currCorners[idx].x, currCorners[idx].y);
            cv::circle(image, pt, 1, {0, 255, 0}, -1, cv::LINE_8);
        }

        for (const auto& idx: unmatchedIndices) {
            cv::Point2d pt(currCorners[idx].x, currCorners[idx].y);
            cv::circle(image, pt, 1, {255, 0, 0}, -1, cv::LINE_8);
        }
    
        std::stringstream ss;
        ss << matchedIndices.size() << "/" << currCorners.size();
        cv::putText(image, ss.str(), {image.cols - 120, image.rows - 120}, 0, 0.8, {0, 255, 0}, 2);

        // Display the image in a window named "Display Window".
        cv::imshow("Display Window", image);
        
        // Wait indefinitely for a key press.
        cv::waitKey(0);

        prevCorners = std::move(currCorners);
        prevDesc = std::move(currDesc);
    }

    return 0;
}
