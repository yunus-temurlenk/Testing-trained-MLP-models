#include <opencv2/opencv.hpp>

int main() {
    // Load the trained model
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::load("/home/cvlab/trained_digit_model.xml");

    // Load the image you want to classify
    cv::Mat testImage = cv::imread("/home/cvlab/Downloads/5.png", cv::IMREAD_GRAYSCALE);  // Assuming the image is grayscale

    cv::resize(testImage,testImage,cv::Size(28,28));
    // Flatten the image
    cv::Mat flattenedImage = testImage.reshape(1, 1);
    cv::Mat input;
    flattenedImage.convertTo(input, CV_32F);

    // Perform prediction
    cv::Mat output;
    mlp->predict(input, output);

    std::cout<<output<<std::endl;
    // Find the class with the highest probability
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

    int predictedClass = classIdPoint.x;

    // Display the result
    std::cout << "Predicted class: " << predictedClass << " with confidence: " << confidence << std::endl;

    return 0;
}
