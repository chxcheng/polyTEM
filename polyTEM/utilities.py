'''
These helper functions are specific to my project and instrumentation in some way
'''
import re
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import jsonpickle.ext.pandas as jsonpickle_pandas
jsonpickle_pandas.register_handlers()

def saveas_json(obj,filename):
    """
    A weakness of pickle is that if I change the class structure after I pickle, I can no longer unpickle it.
    converting to json gets aroudn this while things are debugging
    """
    jsonpickle_numpy.register_handlers()
    jsonpickle_pandas.register_handlers()
    jsonpickle.set_encoder_options('simplejson',allow_nan=False)

    json_str = jsonpickle.encode(obj)
    with open(filename,'w') as f:
        f.write(json_str)
    return

def load_json(filename):
    jsonpickle_numpy.register_handlers()
    jsonpickle_pandas.register_handlers()
    jsonpickle.set_decoder_options('simplejson')
    with open(filename,'r') as f:
        json_str = f.readline()
    obj = jsonpickle.decode(json_str)
    return obj
    

def read_mdoc(filename:str):
    """
    Reads the mdoc file generated during imaging and extracts parameters
    
    Expected file contents:
    T = SerialEM: Glacios 9953405 at Stanford University        26-Oct-23  14:55:45    
    Voltage = 200

    [FrameSet = 0]
    GainReference = CountRef_AM880_2_area_2_0000_0.0.mrc
    TiltAngle = 0.00440576
    StagePosition = -26.4387 53.0248
    StageZ = -71.9289
    Magnification = 36000
    Intensity = 0.456253
    ExposureDose = 0
    DoseRate = 12.9769
    PixelSpacing = 1.151
    SpotSize = 5
    ProbeMode = 0
    Defocus = 0.166593
    ImageShift = 0 0
    RotationAngle = -92.83
    ExposureTime = 3.59521
    Binning = 1
    CameraIndex = 1
    DividedBy2 = 0
    OperatingMode = 1
    UsingCDS = 0
    MagIndex = 31
    LowDoseConSet = 4
    CountsPerElectron = 1
    TargetDefocus = -1
    SubFramePath = \\192.168.42.2\data\Salleo\Christina\20231026_AM880\AM880_2_-1um_area_11_0000.mrc
    NumSubFrames = 18
    FrameDosesAndNumber = 0 18
    DateTime = 26-Oct-2023  17:06:51
    NavigatorLabel = 11
    UncroppedSize = -5760 -4092
    RotationAndFlip = 0
    TimeStamp = 120499611
    
    Args:
        filename: .mrc.mdoc filename
    
    Returns:
        pixel_res: float, angstroms per pixel
        defocus: float, target defocus
    """
    pixel_res_regex = re.compile('(?<=PixelSpacing\s\=\s)-?\d+.?\d+')
    defocus_regex = re.compile('(?<=TargetDefocus\s\=\s)-?\d+.?\d*')
    tilt_regex = re.compile('(?<=TiltAngle\s\=\s)-?\d+.?\d+')
    tilt_angle = 0
    pixel_res = 0
    defocus = 0
    with open(filename,'r') as f:
        for line in f:
            tilt_match = tilt_regex.search(line)
            pixel_match = pixel_res_regex.search(line)
            defocus_match = defocus_regex.search(line)
            if tilt_match:
                tilt_angle = float(tilt_match.group(0))
            elif pixel_match:
                pixel_res = float(pixel_match.group(0))
            elif defocus_match:
                defocus = float(defocus_match.group(0))
            if all([pixel_res,defocus,tilt_angle]):
                break
    
    return pixel_res,defocus,tilt_angle
    
    
        

