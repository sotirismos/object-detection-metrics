## Comparison of various object detection metrics via their precision - recall curves ##

### Usage:

To generate a PR plot of all the metrics for a given threshold of 0.75, dpi scale 200, 12 inch output image size, run:

```bash
python metrics.py --thres 0.75 --dirpath <path to a dir> --pkl <path to .pkl predictions> --dpi 200 --size 12
```

*Arguments*

```
 --dirpath 
```
Absolute or relative path to output directory

```
 --weight 
```
weighting factor for hybrid and fmeasure metrics

```
 --thres
```
decision threshold, a sample with a metric above the threshold is classified as a TP.

```
 --size
```
output plot size in inches

```
 --dpi
```
sets the dpi scale of the plot file

```
 --plot
```
a switch option, show the plot in a window

```
 --pkl
```
a serialized list of dictionaries in pickle file format

```
 --ratio
```
set the aspec ratio of the ouput image
  
---
Each list element is a dictionary that contains information about
the detected and ground truth objects in an image,
a summary of its key/value pairs is presented below:

| Key      | Value |
| -------  | ----- |
| det      | list of detected bboxes in imageai format|
| filename | the filename of the related image |
| gt       | a list of ground truth objects |
| matches  | a list of tupled index pairs |

---
[ImageAI](https://github.com/OlafenwaMoses/ImageAI) returns the following dictionary for each detection in an image:


| Key        |Value  |
| -------    | ----- |
| box_points | bbox corner coordinates in the image's coordinate space [top-left-x, top-left-y, right-bot-x, right-bot-y]|
| name       | the name of the label associated with the detected object |
| percentage_probability | the detected object's confidence score |

---
- Description of matches:

	Let's assume that we tested a detection and classification system on an image resulting in
	no detected objects, although the image contains a ground truth object. 
	Then the set of gt is {0} while the detection set is empty {}.

	```
	[(None, 0)]
	```

	Another case could be that of 2 ground truth objects and 1 detected object.
	In this case the list of tuples are (the second tuple implies a false negative):


	```
	[(0,0), (None,1)]
	```

	An example of FP would be the following:
	```
	[(0, None)]
	```

	Finally, an example of perfect matching, no FP or FN
	```
	[(0,1), (1,0)]
	```
---
### Generation of license plates .pkl prediction files 

- [mdict_list.pkl](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/mdict_list.pkl) is generated running [two_stage_lp.py](https://github.com/sotirismos/GRUBLES-Depersonalization-pipeline/blob/pytorch-mmdetection/two_stage_lp.py) from GRUBLES-Depersonalization-pipeline repository.

- [mdict_list_mmdetection.pkl](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/mdict_list_mmdetection.pkl) is generated running [two_stage_lp_mmdetection.py](https://github.com/sotirismos/GRUBLES-Depersonalization-pipeline/blob/pytorch-mmdetection/two_stage_lp_mmdetection.py) from GRUBLES-Depersonalization-pipeline repository.

---
### Generation of primary features .pkl prediction files 

- [mdict_list_traffic_light.pkl](https://github.com/sotirismos/Object-Detection-Metrics/blob/primary_features_detection_evaluation/mdict_list_traffic_light.pkl) is generated running [inference.py](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/inference.py).

- [mdict_list_traffic_sign.pkl](https://github.com/sotirismos/Object-Detection-Metrics/blob/primary_features_detection_evaluation/mdict_list_traffic_sign.pkl) is generated running [inference.py](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/inference.py).

- [mdict_list_vehicle.pkl](https://github.com/sotirismos/Object-Detection-Metrics/blob/primary_features_detection_evaluation/mdict_list_vehicle.pkl) is generated running [inference.py](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/inference.py).

---
[inference.py](https://github.com/sotirismos/Object-Detection-Metrics/blob/primary_features_detection_evaluation/inference.py) has the following functionalities.
- Loading a pretrained model on BDD100K dataset, which consists of 10 classes.
- Making predictions on images annotated on a subset of BDD100K classes on CVAT and exported as CVAT for images 1.1 (refer to GRUBLES-Annotation-pipeline).
- Edit predictions and return the following dictionary for each detection in an image.

| Key        |Value  |
| -------    | ----- |
| box_points | bbox corner coordinates in the image's coordinate space [top-left-x, top-left-y, right-bot-x, right-bot-y]|
| name       | the name of the label associated with the detected object |
| confidence | the detected object's confidence score |

---
### License plates evaluation plots

ImageAI @ IOU_thres = 0.5            |  mmdetection @ IOU_thres = 0.5
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates_mmdetection/pr_t50.png)

ImageAI @ IOU_thres = 0.6            |  mmdetection @ IOU_thres = 0.6
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates/pr_t60.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates_mmdetection/pr_t60.png)

ImageAI @ IOU_thres = 0.7            |  mmdetection @ IOU_thres = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates/pr_t70.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates_mmdetection/pr_t70.png)

---
### Primary features evaluation plots

Traffic Light @ IOU_thres = 0.5            |  Traffic Light @ IOU_thres = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_light/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_light/pr_t70.png)

Traffic Sign @ IOU_thres = 0.5            |  Traffic Sign @ IOU_thres = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_sign/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_light/pr_t70.png)

Vehicle @ IOU_thres = 0.5            |  Vehicle @ IOU_thres = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_vehicle/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_vehicle/pr_t70.png)
