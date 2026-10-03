#include "MVS.hpp"

#include <MvCameraControl.h>

#include <cstring>

#include <opencv2/imgproc.hpp>

namespace {

struct SetParam {
    int width = 1440;
    int height =1080;
    float exp_time = 10000.0F;
    float gain = 0.0F;
    float gamma = 1.0F;
};

struct MvsCamera {
    void* handle = nullptr;
    bool sdk_initialized = false;
    bool device_opened = false;
    bool grabbing = false;
    MV_CC_DEVICE_INFO device{};
    SetParam param{};
};

MvsCamera& camera() {
    static MvsCamera value;
    return value;
}

const char* serialNumber(const MV_CC_DEVICE_INFO& device) {
    if (device.nTLayerType == MV_USB_DEVICE) {
        return reinterpret_cast<const char*>(device.SpecialInfo.stUsb3VInfo.chSerialNumber);
    }
    return nullptr;
}

bool selectCamera() {
    MvsCamera& c = camera();

    if (!c.sdk_initialized) {
        if (MV_CC_Initialize() != MV_OK) {
            return false;
        }
        c.sdk_initialized = true;
    }

    MV_CC_DEVICE_INFO_LIST list{};
    if (MV_CC_EnumDevices(MV_USB_DEVICE, &list) != MV_OK) return false;
    

    for (unsigned int i = 0; i < list.nDeviceNum; ++i) {
        MV_CC_DEVICE_INFO* device = list.pDeviceInfo[i];
        if (device == nullptr) {
            continue;
        }

        const char* serial = serialNumber(*device);
        if (serial != nullptr && std::strcmp(serial, "00D36741056") == 0) {
            c.device = *device;
            return true;
        }
    }

    return false;
}

bool openCamera() {
    MvsCamera& c = camera();

    if (c.device_opened) return true;
    
    if (!selectCamera()) return false;

    if (MV_CC_CreateHandle(&c.handle, &c.device) != MV_OK) {
        return false;
    }

    if (MV_CC_OpenDevice(c.handle) != MV_OK) {
        MV_CC_DestroyHandle(c.handle);
        c.handle = nullptr;
        return false;
    }

    c.device_opened = true;
    return true;
}

void setCameraParams() {
    MvsCamera& c = camera();
    MV_CC_SetIntValueEx(c.handle, "Width", c.param.width);
    MV_CC_SetIntValueEx(c.handle, "Height", c.param.height);
    MV_CC_SetEnumValue(c.handle, "TriggerMode", 0);
    MV_CC_SetFloatValue(c.handle, "ExposureTime", c.param.exp_time);
    MV_CC_SetFloatValue(c.handle, "Gain", c.param.gain);
    MV_CC_SetFloatValue(c.handle, "Gamma", c.param.gamma);
}

bool startCamera() {
    MvsCamera& c = camera();

    if (c.grabbing) {
        return true;
    }
    if (!c.device_opened) {
        return false;
    }

    setCameraParams();
    if (MV_CC_StartGrabbing(c.handle) != MV_OK) {
        return false;
    }

    c.grabbing = true;
    return true;
}

}  // namespace

namespace MVS {

bool getframe(cv::Mat& pic) {
    pic.release();
    MvsCamera& c = camera();

    if (!openCamera() || !startCamera()) {
        return false;
    }

    MV_FRAME_OUT frame{};
    if (MV_CC_GetImageBuffer(c.handle, &frame, 400) != MV_OK) return false;
    

    cv::Mat raw(
        static_cast<int>(frame.stFrameInfo.nHeight),
        static_cast<int>(frame.stFrameInfo.nWidth),
        CV_8UC1,
        frame.pBufAddr);


    
    if (!raw.empty()) {
        cv::cvtColor(raw, pic, cv::COLOR_BayerRG2BGR);
        cv::flip(pic, pic, -1);
    }

    const int result = MV_CC_FreeImageBuffer(c.handle, &frame);
    return result == MV_OK && !pic.empty();
}

void shutdown() {
    MvsCamera& c = camera();

    if (c.grabbing) {
        MV_CC_StopGrabbing(c.handle);
        c.grabbing = false;
    }
    if (c.device_opened) {
        MV_CC_CloseDevice(c.handle);
        c.device_opened = false;
    }
    if (c.handle != nullptr) {
        MV_CC_DestroyHandle(c.handle);
        c.handle = nullptr;
    }
    if (c.sdk_initialized) {
        MV_CC_Finalize();
        c.sdk_initialized = false;
    }
}

} 
