#include "opencv2/opencv.hpp"

#define DEG2RAD(v)  (v * CV_PI / 180)
#define Rx(rx)      (cv::Matx33d(1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx)))
#define Ry(ry)      (cv::Matx33d(cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry)))
#define Rz(rz)      (cv::Matx33d(cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1))

class MouseDrag
{
public:
    MouseDrag() : dragged(false) { }
    bool dragged;
    cv::Point start, end;
};

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    if (param == NULL) return;
    auto drag = static_cast<MouseDrag*>(param);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        drag->dragged = true;
        drag->start = cv::Point(x, y);
        drag->end = cv::Point(0, 0);
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (drag->dragged) drag->end = cv::Point(x, y);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        if (drag->dragged)
        {
            drag->dragged = false;
            drag->end = cv::Point(x, y);
        }
    }
}

struct CamParam {
    double f, cx, cy;
    using rotation3d = cv::Point3d;
    using translation3d = cv::Point3d;
    rotation3d rot; // rpy
    translation3d tx; // xyz
};

int main()
{
    const char* input = "data/daejeon_station.png";
    CamParam camera;
    camera.f = 810.5;
    camera.cx = 480;
    camera.cy = 270;
    camera.rot = {DEG2RAD(-18.7), DEG2RAD(-8.2), DEG2RAD(2.0)};
    camera.tx = {0.,-3.31,0.}; // aligned to camera coord.:
                               //  x --> +
                               // y
                               // | z:X
                               // v
                               // +

    cv::Range grid_x(-2, 3), grid_z(5, 35);

    // Load an images
    cv::Mat image = cv::imread(input);
    if (image.empty()) return -1;

    // Configure mouse callback
    MouseDrag drag;
    cv::namedWindow("3DV Tutorial: Object Localization and Measurement");
    cv::setMouseCallback("3DV Tutorial: Object Localization and Measurement", MouseEventHandler, &drag);

    // Draw grids on the ground
    // K: calibration matrix
    // Rc: camera rotation
    // Rg: inverse(Rc)
    // tg: - inverse(Rc) * tc
    cv::Matx33d K(camera.f, 0, camera.cx, 0, camera.f, camera.cy, 0, 0, 1);

    cv::Matx33d Rc = Rz(camera.rot.z) * Ry(camera.rot.y) * Rx(camera.rot.x);
    cv::Matx33d Rg = Rc.t();
    using Translation3d = cv::Point3d;
    Translation3d tx = camera.tx;
    Translation3d tg = -Rg * tx;

    for (int z = grid_z.start; z <= grid_z.end; z++)
    {
        // project ground line ends to camera coord.
        cv::Point3d p = K * (Rg * cv::Point3d(grid_x.start, 0, z) + tg);
        cv::Point3d q = K * (Rg * cv::Point3d(grid_x.end, 0, z) + tg);
        cv::line(image, cv::Point2d(p.x / p.z, p.y / p.z), cv::Point2d(q.x / q.z, q.y / q.z), cv::Vec3b(64, 128, 64), 1);
    }
    for (int x = grid_x.start; x <= grid_x.end; x++)
    {
        // project ground line ends to camera coord.
        cv::Point3d p = K * (Rg * cv::Point3d(x, 0, grid_z.start) + tg);
        cv::Point3d q = K * (Rg * cv::Point3d(x, 0, grid_z.end) + tg);
        cv::line(image, cv::Point2d(p.x / p.z, p.y / p.z), cv::Point2d(q.x / q.z, q.y / q.z), cv::Vec3b(64, 128, 64), 1);
    }

    while (true)
    {
        cv::Mat image_copy = image.clone();
        if (drag.end.x > 0 && drag.end.y > 0)
        {
            // Calculate object location and height
            // unit is pixel
            //
            // pt in cam coord.
            cv::Point3d pt = cv::Point3d(drag.start.x - camera.cx, drag.start.y - camera.cy, camera.f);
            // pt in world coord.
            pt = Rc * pt; // camera has only rotation. center aligned to (0,0,0)

            if (pt.y < DBL_EPSILON) 
                continue; // Skip the degenerate case (beyond the horizon)

            // pt' in cam coord.
            cv::Point3d pt_prime = cv::Point3d(drag.end.x - camera.cx, drag.end.y - camera.cy, camera.f);
            // pt' in world coord.
            pt_prime = Rc * pt_prime;

            if (pt_prime.y < DBL_EPSILON) 
                continue; // Skip the degenerate case (beyond the horizon)

            double L = -camera.tx.y;
            
            double X = pt.x / pt.y * L; // not needed, but true for any point, true for pt
            double Z = pt.z / pt.y * L; // true for any point, true for pt
            double Z_prime = pt_prime.z / pt_prime.y * L; // true for pt_prime too
            double H = L * (Z_prime - Z) / Z_prime; // assuming obj is orthorgonal to ground plane

            // Draw head/contact points and location/height
            cv::line(image_copy, drag.start, drag.end, cv::Vec3b(0, 0, 255), 2);
            cv::circle(image_copy, drag.end, 4, cv::Vec3b(255, 0, 0), -1);
            cv::circle(image_copy, drag.start, 4, cv::Vec3b(0, 255, 0), -1);
            cv::putText(image_copy, cv::format("X:%.2f, Z:%.2f, H:%.2f", X, Z, H), drag.start + cv::Point(-20, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 255, 0));
        }

        // Show the image
        cv::imshow("3DV Tutorial: Object Localization and Measurement", image_copy);
        int key = cv::waitKey(1);
        if (key == 27) break; // 'ESC' key: Exit
    }
    return 0;
}
