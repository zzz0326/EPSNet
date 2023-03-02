# gunzip location
MainPath = '***'

# Input image and saliency map size
INPUT_SIZE = (320, 160)

# encoder training data
pathToEncoderTrain = MainPath + '/proxy_task/'

# decoder training data
pathToDecoderInput = MainPath + '/VR-EyeTracking/input/'
pathToDecoderFixationMap = MainPath + '/VR-EyeTracking/fixation_map/'
pathToDecoderSaliencyMap = MainPath + '/VR-EyeTracking/saliency/'

# val data
pathToValInput = MainPath + '/Salient360/input/'
pathToValFixationMap = MainPath + '/Salient360/fixmap/'
pathToValSaliencyMap = MainPath + '/Salient360/saliency/'