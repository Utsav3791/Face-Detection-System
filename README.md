# Face Detection System
Face Detection System using libraries such as OpenCV, NumPy and Pandas
## Introduction
Face detection has gained a lot of attention due to its real-time applications. A lot of research has been done and still going on for improved and fast implementation of the face detection algorithm. 

__Why is face detection difficult for a machine?__ 
Face detection is not as easy as it seems due to lots of variations of image appearance, such as pose variation (front, non-front), occlusion, image orientation, illumination changes and facial expression.

### OpenCV

OpenCV is an open source computer vision and machine learning software library. It is a BSD-licence product thus free for both business and academic purposes.The Library provides more than 2500 algorithms that include machine learning tools for classification and clustering, image processing and vision algorithm, basic algorithms and drawing functions, GUI and I/O functions for images and videos. Some applications of these algorithms include face detection, object recognition, extracting 3D models, image processing, camera calibration, motion analysis etc.

OpenCV is written natively in C/C++. It has C++, C, Python and Java interfaces and supports Windows, Linux, Mac OS, iOS, and Android. OpenCV was designed for computational efficiency and targeted for real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing.

For face detection specifically, there are two pre-trained classifiers:

1. Haar Cascade Classifier
2. LBP Cascade Classifier


We will explore both face detectors in this tutorial. 

### Haar Cascade Classifier

It is a machine learning based approach where a cascade function is trained from a lot of positive (images with face) and negative images (images without face). The algorithm is proposed by Paul Viola and Michael Jones.

The algorithm has four stages:

1. **Haar Feature Selection:** Haar features are calculated in the subsections of the input image. The difference between the sum of pixel intensities of adjacent rectangular regions is calculated to differentiate the subsections of the image. A large number of haar-like features are required for getting facial features.
2. **Creating an Integral Image:** Too much computation will be done when operations are performed on all pixels, so an integral image is used that reduce the computation to only four pixels. This makes the algorithm quite fast.
3. **Adaboost:** All the computed features are not relevant for the classification purpose. `Adaboost` is used to classify the relevant features.
4. **Cascading Classifiers:** Now we can use the relevant features to classify a face from a non-face but algorithm provides another improvement using the concept of `cascades of classifiers`. Every region of the image is not a facial region so it is not useful to apply all the features on all the regions of the image. Instead of using all the features at a time, group the features into different stages of the classifier.Apply each stage one-by-one to find a facial region. If on any stage the classifier fails, that region will be discarded from further iterations. Only the facial region will pass all the stages of the classifier.   

### LBP Cascade Classifier

LBP is a texture descriptor and face is composed of micro texture patterns. So LBP features are extracted to form a feature vector to classify a face from a non-face. Following are the basic steps of LBP Cascade classifier algorithm:

1. **LBP Labelling:** A label as a string of binary numbers is assigned to each pixel of an image.
2. **Feature Vector:** Image is divided into sub-regions and for each sub-region, a histogram of labels is constructed. Then, a feature vector is formed by concatenating the sub-regions histograms into a large histogram.
3. **AdaBoost Learning:** Strong classifier is constructed using gentle AdaBoost to remove redundant information from feature vector.
4. **Cascade of Classifier:** The cascades of classifiers are formed from the features obtained by the gentle AdaBoost algorithm. Sub-regions of the image is evaluated starting from simpler classifier to strong classifier. If on any stage classifier fails, that region will be discarded from further iterations. Only the facial region will pass all the stages of the classifier.

### Comparison between Haar and LBP Cascade Classifier

A short comparison of `haar cascade classifier` and `LBP cascade classifier` is given below :

<TABLE  BORDER="1">
  
   <TR>
      <TH>Algorithm</TH>
      <TH>Advantages</TH>
      <TH>Disadvantages</TH>
   </TR>
   <TR>
      <TD>Haar </TD>
      <TD>
      <ol>
        <li>High detection accuracy</li>
        <li>Low false positive rate</li>
      </ol>
      </TD>
      <TD>
      <ol>
        <li>Computationally complex and slow</li>
        <li>Longer training time</li>
        <li>Less accurate on black faces</li>
        <li>Limitations in difficult lightening conditions</li>
        <li>Less robust to occlusion</li>
      </ol>
      </TD>
   </TR>
   <TR>
      <TD>LBP</TD>
      <TD>
      <ol>
        <li>Computationally simple and fast</li>
        <li>Shorter training time</li>
        <li>Robust to local illumination changes</li>
        <li>Robust to occlusion</li>
      </ol>
      </TD>
      <TD>
      <ol>
        <li>Less accurate</li>
        <li>High false positive rate</li>
      </ol>
      </TD>
   </TR>
</TABLE>

Each OpenCV face detection classifier has its own pros and cons but the major differences are in accuracy and speed. So in a use case where more accurate detections are required, `Haar` classifier is more suitable like in security systems, while `LBP` classifier is faster than Haar classifier and due to its fast speed, it is more preferable in applications where speed is important like in mobile applications or embedded systems. 


### About My Code:


__1.Frame Differnce__: 
I created a mask for every frame which shows parts of the image that are in motion.Following is the code for computing frame differences:
```python
def frame_diff(prev_frame, cur_frame, next_frame):

    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)

    return cv2.bitwise_and(diff_frames_1, diff_frames_2)
```
then for getting current frame from webcam is:
```python
def get_frame(cap, scaling_factor):
    _, frame = cap.read()

    frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return gray
```
__2. Colour Spaces__: 
There are more than 150 color-space conversion methods available in OpenCV. But we will look into only two, which are most widely used ones: BGR ↔ Gray and BGR ↔ HSV. For more info, [click here](https://github.com/Utsav3791/Face-Detection-System/blob/main/02%20(ColourSpaces).py)

__3. Background Subtraction__: 
Built bakground subtration using OpenCV and created object named `bg_subtractor`. [code](https://github.com/Utsav3791/Face-Detection-System/blob/main/03%20(Background%20Subtraction).py)

__4.Camshift__:
Camshift or we can say Continuously Adaptive Meanshift is an enhanced version of the meanshift algorithm which provides more accuracy and robustness to the model. With the help of Camshift algorithm, the size of the window keeps updating when the tracking window tries to converge. The tracking is done by using the color information of the object. Also, it provides the best fitting tracking window for object tracking. It applies meanshift first and then updates the size of the window as:

<a href="https://www.codecogs.com/eqnedit.php?latex=s&space;=&space;2\times\sqrt{\frac{M_{00}}{256}}\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s&space;=&space;2\times\sqrt{\frac{M_{00}}{256}}\" title="s = 2\times\sqrt{\frac{M_{00}}{256}}\" /></a>
   
It then calculates the best fitting ellipse to it and again applies the meanshift with the newly scaled search window and the previous window. This process is continued until the required accuracy is met.
[Click here for code](https://github.com/Utsav3791/Face-Detection-System/blob/main/04%20(CamShift).py)

__5. Face Detector__:
Face detection using Haar cascades is a machine learning based approach where a cascade function is trained with a set of input data. OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc..

You need to download the trained classifier XML file (haarcascade_frontalface_default.xml), which is available in OpenCv’s GitHub repository. Save it to your working location.

To load and check haar cascade file has been loaded correctly or not:
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')
```
for initializing video capture:
```python
cap = cv2.VideoCapture(0)
```
[Click here for code](https://github.com/Utsav3791/Face-Detection-System/blob/main/05%20(Face%20Detector).py)

__6. Motion Detector__:
Now we will see what are the phases that we must follow to create an algorithm that allows us to detect movement with OpenCV. The process will perform various tasks:
1. Grayscale conversion and noise removal
2. Subtraction operation between the background and the foreground.
3. Apply a threshold to the image resulting from the subtraction.
4. Detection of contours or blobs

for more highlights, [Click here for code](https://github.com/Utsav3791/Face-Detection-System/blob/main/06%20(Motion%20Detector).py)

### Applications
* Face Identification
* Access Control
* Security
* Image database investigations
* General identity verification
* Surveillance

### Requirements:
[numpy](https://pypi.org/project/numpy/)\
[pandas](https://pypi.org/project/pandas/)\
[opencv](https://pypi.org/project/opencv-python/)

__OR__

download `requirements.txt` file and run:
>>pip install -r requirements.txt

### Some Commands to use:
* For checking python version you have 
>>python --version

* To install the latest version of a package:
>>pip install PackageName

* To install a specific version, type the package name followed by the required version:
>>pip install 'PackageName==1.4'

* To upgrade an already installed package to the latest from PyPI:
>>pip install --upgrade PackageName

* Uninstalling/removing a package is very easy with pip:
>>pip uninstall PackageName

