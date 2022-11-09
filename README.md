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
### Metrics
In order to evaluate the performance of the models for each class (license plates, traffic lights, traffic signs, vehicles) we calculated the mAP based on the
precision-recall curves and exploting various metrics on different threshold values [0.5, 0.7]. To get the TP, FP, FN in order to construct the precision-recall curves, we utilized the following metrics.
- IoU = intersection_area / union_area
- "Precision" = intersection_area / detection_area
- "Recall" = intersection_area / gt_area
- "LP metric 1" = detection_area / union_area
- "LP metric 2" = gt_area / union_area

![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/metrics.jpg)

---
### Evaluation datasets
For the class of license plates, the evaluation dataset is a subset of the Stanford cars dataset and consists of 105 images of different types of vehicles (mainly cars) accompanied by 105 license plate annotation files. <br />
For the classes of vehicles (cars, buses, trucks, motorcycles), traffic lights and traffic signs the dataset collected during the GRUBLES projects was utilized. More specifically, videos from the 6 different Insta360 Pro2 cameras were randomly selected. The frames from these videos were extracted and 500 random frames were selected, where each frame was annotated at a bounding box level, resulting to 1616 annotated vehicles, 125 annotated traffic lights and 371 annotated traffic signs.

---
### License plates evaluation plots

ImageAI @ threshold = 0.5            |  mmdetection @ threshold = 0.5
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates_mmdetection/pr_t50.png)

ImageAI @ threshold = 0.6            |  mmdetection @ threshold = 0.6
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates/pr_t60.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates_mmdetection/pr_t60.png)

ImageAI @ threshold = 0.7            |  mmdetection @ threshold = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates/pr_t70.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_license_plates_mmdetection/pr_t70.png)

---
### Primary features evaluation plots

Traffic Light @ threshold = 0.5            |  Traffic Light @ threshold = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_light/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_light/pr_t70.png)

Traffic Sign @ threshold = 0.5            |  Traffic Sign @ threshold = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_sign/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_traffic_sign/pr_t70.png)

Vehicle @ threshold = 0.5            |  Vehicle @ threshold = 0.7
:-------------------------:|:-------------------------:
![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_vehicle/pr_t50.png)  |  ![](https://github.com/sotirismos/Object-Detection-Metrics/blob/master/plots_vehicle/pr_t70.png)

---
### Conclusion
The two-stage license plate detection approach utilizing [mmdetection](https://github.com/sotirismos/mmdetection) provides much better perfomance that the [ImageAI](https://github.com/OlafenwaMoses/ImageAI) approach. <br />
For the class of traffic lights, during the annotation process for a specific traffic light we outline 1 bounding box, while during the prediction process this can be "broken" into 2 or more, as shown in the figure below. Also, during prediction inverted traffic lights that have not been annotated are predicted as *True Positives*. Setting the threshold for IoU to 0.5, the mAP results to 40.69%. This is justified by the fact that there are many cases where we have 2 or more predictions and 1 annotation, where the predictions have a very small IoU and as a result are categorized as *False Positives*, and the prediction of inverted traffic lights as *True Positives*. Thus, the number of *False Positives* is enormous and the mAP is low. <br />
For the vehicles the model works perfectly. Setting the threshold for IoU equal to 0.5 results to mAP = 62.28%, which is due to the fact that vehicles were not exhaustively annotated, resulting in a high number of *False Positives*. The ability of the model to locate almost every vehicle without fail is illustrated below and is verified by the fact that the Recall metric for this class is equal to 1. <br />
For the class of traffic signs, the model works satisfactorily. Setting the threshold for the IoU equal to 0.5 results to mAP = 34.71%, where this particular low value is due to the fact that the model also detects the inverted signs that we did not annotate resulting in a high number of *False Positives*, as illustrated below.
