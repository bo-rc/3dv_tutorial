#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{
    cv::String keys =
        "{help h usage ? |      | print this message   }"
        "{video  | | video path}"     // optional, video path value ""
        "{cam  | | webcam id}";       // optional, default value ""


    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    bool hasVideo = parser.has("video");
    bool hasCam = parser.has("cam");
    cv::String video_path = parser.get<cv::String>("video"); 
    int cam_id = parser.get<int>("cam");


    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    // Open a stream
    std::unique_ptr<cv::VideoCapture> streamPtr;
    streamPtr = nullptr;
    if (hasVideo) {
        streamPtr = std::make_unique<cv::VideoCapture>(video_path);
    } else if (hasCam) {
        streamPtr = std::make_unique<cv::VideoCapture>(cam_id);
    } else {
        return 0;
    }

    cv::Matx33d K(432.7390364738057, 0, 476.0614994349778, 0, 431.2395555913084, 288.7602152621297, 0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };
    cv::Size board_pattern(10, 7);
    double board_cellsize = 0.025;
    // cv::Size board_pattern(9, 6);
    // double board_cellsize = 0.020;

    // Prepare a 3D box for simple AR
    std::vector<cv::Point3d> box_lower = { 
        cv::Point3d(4 * board_cellsize, 2 * board_cellsize, 0), 
        cv::Point3d(5 * board_cellsize, 2 * board_cellsize, 0), 
        cv::Point3d(5 * board_cellsize, 4 * board_cellsize, 0), 
        cv::Point3d(4 * board_cellsize, 4 * board_cellsize, 0) };
    std::vector<cv::Point3d> box_upper = { 
        cv::Point3d(4 * board_cellsize, 2 * board_cellsize, -board_cellsize), // points already in camera coord.
        cv::Point3d(5 * board_cellsize, 2 * board_cellsize, -board_cellsize), 
        cv::Point3d(5 * board_cellsize, 4 * board_cellsize, -board_cellsize), 
        cv::Point3d(4 * board_cellsize, 4 * board_cellsize, -board_cellsize) };

    // Prepare 3D points on a chessboard
    std::vector<cv::Point3d> obj_points;
    for (int r = 0; r < board_pattern.height; r++)
        for (int c = 0; c < board_pattern.width; c++)
            obj_points.push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));

    // Run pose estimation
    while (streamPtr->isOpened())
    {
        // Grab an image from the video
        cv::Mat image;
        *streamPtr >> image;
        if (image.empty()) break;

        // Estimate camera pose
        std::vector<cv::Point2d> img_points;
        bool success = cv::findChessboardCorners(image, board_pattern, img_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (success)
        {
            cv::Mat rvec, tvec;
            cv::solvePnP(obj_points, img_points, K, dist_coeff, rvec, tvec);

            // Draw the box on the image
            cv::Mat line_lower, line_upper;
            cv::projectPoints(box_lower, rvec, tvec, K, dist_coeff, line_lower);
            cv::projectPoints(box_upper, rvec, tvec, K, dist_coeff, line_upper);
            line_lower.reshape(1).convertTo(line_lower, CV_32S); // Change 4 x 1 matrix (CV_64FC2) to 4 x 2 matrix (CV_32SC1)
            line_upper.reshape(1).convertTo(line_upper, CV_32S); // Because 'cv::polylines()' only accepts 'CV_32S' depth.
            cv::polylines(image, line_lower, true, cv::Vec3b(255, 0, 0), 2);
            for (int i = 0; i < line_lower.rows; i++)
                cv::line(image, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Vec3b(0, 255, 0), 2);
            cv::polylines(image, line_upper, true, cv::Vec3b(0, 0, 255), 2);

            // Print camera position
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            cv::Mat p = -R.t() * tvec;
            std::cout << p << std::endl;
            // std::cout << R << std::endl;
            // cv::String info = cv::format("XYZ: [%.3f, %.3f, %.3f]", cv::Point3d(p));
            cv::String info = cv::format("XYZ: [%.3f, %.3f, %.3f]   Rotation Vec: [%.3f, %.3f, %.3f]", cv::Point3d(p).x, cv::Point3d(p).y, cv::Point3d(p).z, cv::Point3d(rvec).x, cv::Point3d(rvec).y, cv::Point3d(rvec).z);
            cv::putText(image, info, cv::Point(10, 22), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Vec3b(0, 255, 0));
        }

        // Show the image
        cv::imshow("3DV Tutorial: Pose Estimation (Chessboard)", image);
        int key = cv::waitKey(1);
        if (key == 27) break; // 'ESC' key: Exit
    }

    streamPtr->release();
    return 0;
}
