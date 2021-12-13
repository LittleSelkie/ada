from detection_algorithms.outlier_detection import *
from detection_algorithms.outlier_detection import load_mydetector as load_detec

# current training path contains 100 images and takes ~ 3m30s

study_dir = "D:\Video\Skyrim\Screens\Training\\**\*.*"

detector_dir = "D:\Video\Skyrim\detector\\"

test1 = "D:\Video\Skyrim\Screens\Outliers\Test1\\**\*.*"
test2 = "D:\Video\Skyrim\Screens\Outliers\Test2\\**\*.*"
test4 = "D:\Video\Skyrim\Screens\Outliers\Test4\\**\*.*"
test5 = "D:\Video\Skyrim\Screens\Outliers\Test5\\**\*.*"

load_detec(detector_dir, 0) # 1 - new learning path; 0 - no learning
find(test1, detector_dir, 0)
find(test2, detector_dir, 0)
find(test4, detector_dir, 0)
find(test5, detector_dir, 0)