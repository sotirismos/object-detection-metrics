# A comparison of various object detection metrics via their precision - recall curves
---

## Usage:

To generate a PR plot of all the metrics for a given threshold of 0.7, weighting factor 0.5,
dpi scale 300, 12 inch output image size, run:

```bash
python metrics.py --weight 0.5 --thres 0.7 --dirpath <path to a dir> --dpi 300  --size 12
```

*Arguments*

```
 --dirpath 
```
: Absolute or relative path to output directory

```
 --weight 
```
: weighting factor for hybrid and fmeasure metrics

```
 --thres
```
: decision threshold, a sample with a metric above the threshold is classified as a TP.

```
 --size
```
: output plot size in inches

```
 --dpi
```
: sets the dpi scale of the plot file

```
 --plot
```
: a switch option, show the plot in a window

```
 --pkl
```
: a serialized list of dictionaries in pickle file format

```
 --ratio
```
: set the aspec ratio of the ouput image
  
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
ImageAI returns the following dictionary for each detection in an image:

| Key        |Value| |
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
## Example plot

![plot](https://bitbucket.org/datascouting/detection_metrics/raw/master/plots/pr_t50_w80.png)
