import pickle
import enum
import copy
import os
import argparse
import numpy as np
import itertools as it
from matplotlib import pyplot
from matplotlib import gridspec

class Errors(enum.Enum):
    """
    A enum class used as a flag of FN, FP
    samples in the numpy arrays of the metrics
    """
    FN = -1
    FP = -2

def match_gen(iou, step, n):
    """
    A generator function that yields a tuple of matching indices between the
    sets of detected bboxes and ground truth bboxes. Only non zero IoU values
    are taken into account for pair matching.

    Parameters
    ----------
    iou : numpy array
        A numpy array that contains the IoU values
        of the cartesian product between the detected and ground truth bbox sets.
    step : int
        The cardinality of the ground truth bboxes set.
    n : int
        The cardinality of the detected bboxes set.

    Yields
    ------
    pair : tuple
        The matcing pair of detected and ground truth bbox indices
        (in their respective set).

    """
    iou_m = np.ma.array(iou, mask=False) # a masked array view of the iou array
    #iou_m = np.array(iou)
    
    for i in range(n):
        try: # exception handling of an out of bounds index
            iou_view = iou_m[step*i:step*(i+1)]# ith det bbox pairs in the cart prod set

            # gt bbox index of the pair with the largest IoU among all non-zero and non-masked pairs
            ind = iou_view.nonzero()[0][iou_view[iou_view>0].argmax()]
	    #ind = np.where(iou_view == max(iou_view[iou_view>0]))[0][0]

            # mask the selected index
            iou_m.mask[ [step*i + ind for i in range(1,n)] ] = True

            yield (i, ind) # yield the pair (index of det bbox, index of gt bbox)
        except ValueError:
            pass

def match(det_obj, tr_bboxes):
    """
    This function matches det bboxes to gt bboxes.

    Parameters
    ----------
    det_obj : ImageAI detection object
        A dictionary of the det bbox label name, condfidence value and box points.
            Key                         Value type
            ----                        ------
            name                        string
            percentage_probability      double  (range in 0-100)
            box_points                  list of uints (length 4)
    tr_bboxes : a numpy array of uints
        A 4 element array that contains a bbox top-left corner x,y and 
        bottom-right corner x,y values.

    Returns
    -------
    mpair list : a list of tuples
        A list of tuples representing the det and gt bbox index pairs.

    """
    cart_prod = it.product(det_obj, tr_bboxes) # cartesian product of the two bbox sets

    # IoU values of the bbox pairs using list comprehension
    iou = [ calculate_iou(pair[1], pair[0]['box_points']) for pair in cart_prod ]

    mpair_list = [ mpair for mpair in match_gen(iou, len(tr_bboxes), len(det_obj)) ]

    diff = lambda a,b: np.uint(np.setdiff1d(np.union1d(a,b), np.intersect1d(a,b)))

    det_list = list()
    gt_list = list()

    if det_obj:
        if mpair_list:
            matched_det = [mpair[0] for mpair in mpair_list if mpair]
            det_list += [ (ind, None) for ind in diff( matched_det, range(len(det_obj)) ) ]
        else:
            det_list += [ (ind, None) for ind in range(len(det_obj)) ]

    if tr_bboxes:
        if mpair_list:
            matched_gt = [mpair[1] for mpair in mpair_list if mpair]
            gt_list += [ (None, ind) for ind in diff( matched_gt, range(len(tr_bboxes)) ) ]
        else:
            gt_list += [ (None, ind) for ind in range(len(tr_bboxes)) ]

    mpair_list += (det_list + gt_list)
    return mpair_list

def calculate_areas(bbox_det, bbox_gt):
    """
    Helper function.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detectio bounding box
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bounding box
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    area_det : numpy scalar of type float
        Area of detection bounding box.
    area_gt : numpy scalar of type float
        Area of ground truth bounding box.
    intersection_area: numpy scalar of type float
        Intersection area of the two bounding boxes.
    """
    bbox_det = np.asarray(bbox_det) # ensure numpy array
    bbox_gt = np.asarray(bbox_gt) # ensure numpy array

    assert len(bbox_det) == 4, "calculate_iou: Bbox_det length is %d" % len(bbox_det)
    assert len(bbox_gt) == 4, "calculate_iou: Bbox_gt length is %d" % len(bbox_gt)

    tl1 = bbox_det[:2] # top-left corner x,y pair detection
    tl2 = bbox_gt[:2] # top-left corner x,y pair of ground truth
    br1 = bbox_det[2:] # bottom-right corner x,y pair of detection
    br2 = bbox_gt[2:] # bottom-right corner x,y pair of ground truth

    # width and height of the resulting box after the intersection
    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)

    # area of the resulting box after the intersection
    intersection_area = np.prod(intersection_wh)

    # the areas of the det bboxes and gt bboxes
    area_det, area_gt = np.prod(np.abs(br1 - tl1)), np.prod(np.abs(br2 - tl2))

    return area_det, area_gt, intersection_area

def calculate_iou(bbox_det, bbox_gt):
    """
    This function calculates the intersection over union of 
    given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    iou : numpy scalar of type float
        The IoU value.
    """
    
    # calculate detection bbox, ground truth and their intersection areas
    area_det, area_gt, intersection_area = calculate_areas(bbox_det, bbox_gt)

    # union area
    union_area = area_det + area_gt - intersection_area

    return intersection_area / union_area

def calculate_recall(bbox_det, bbox_gt):
    """
    Calculate the recall given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    recall : numpy scalar of type float
        The Recall value.

    """
    # calculate detection bbox, ground truth and their intersection areas
    _, area_gt, intersection_area = calculate_areas(bbox_det, bbox_gt)

    return intersection_area / area_gt

def calculate_precision(bbox_det, bbox_gt):
    """
    Calculate the precision metric given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    precision : numpy scalar of type float
        The Precision value.

    """
    # calculate detection bbox, ground truth and their intersection areas
    area_det, _, intersection_area = calculate_areas(bbox_det, bbox_gt)

    return intersection_area / area_det

def calculate_lpmetric1(bbox_det, bbox_gt):
    """
    Calculate lpmetric1 given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    lpmetric1 : numpy scalar of type float
        The lpmetric value.
    """
    # calculate detection bbox, ground truth and their intersection areas
    area_det, area_gt, intersection_area = calculate_areas(bbox_det, bbox_gt)

    # union area
    union_area = area_det + area_gt - intersection_area

    return area_det / union_area

def calculate_lpmetric2(bbox_det, bbox_gt):
    """
    Calculate lpmetric given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].

    Returns
    -------
    lpmetric2 : numpy scalar of type float
        The lpmetric value.
    """
    # calculate detection bbox, ground truth and their intersection areas
    area_det, area_gt, intersection_area = calculate_areas(bbox_det, bbox_gt)

    # union area
    union_area = area_det + area_gt - intersection_area

    return area_gt / union_area

def calculate_hybrid(bbox_det, bbox_gt, weight=0.5):
    """
    Calculate the hybrid metric given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    weight : a double value, optional
        The weighting parameter of the linear combination of recall and iou metrics.
        The default is 0.5

    Returns
    -------
    hybrid : numpy scalar of type float
        The weighted linear combination of recall and iou metrics.
    """
    if (weight <= 0 or weight >= 1):
        raise ValueError("Weight parameter in hybrid metric must be in range (0,1)")

    weight = min(max(weight, 0), 1)

    recall = calculate_recall(bbox_det, bbox_gt)
    iou = calculate_iou(bbox_det, bbox_gt)

    return weight * recall + (1 - weight) * iou

def calculate_lp_comb(bbox_det, bbox_gt, weight=0.5):
    """
    A combined metric based on the linear combination of lpmetric2 and
    precision.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    weight : a double value, optional
        The weighting parameter of the linear combination of recall and iou metrics.
        The default is 0.5

    Returns
    -------
    lp_comb : numpy scalar of type float
        The weighted linear combination of recall and iou metrics.
    """
    if (weight <= 0 or weight >= 1):
        raise ValueError("Weight parameter in hybrid metric must be in range (0,1)")

    weight = min(max(weight, 0), 1)

    lpmetric2 = calculate_lpmetric2(bbox_det, bbox_gt)
    precision = calculate_precision(bbox_det, bbox_gt)

    return weight * lpmetric2 + (1 - weight) * precision

def calculate_fmeasure(bbox_det, bbox_gt, weight=0.5):
    """
    F-measure based metric, the weighted harmonic mean of IoU
    and Recall, given a detection and a ground truth bounding box.

    Parameters
    ----------
    bbox_det : a list of 4 elements or a numpy array of 4 elements
        Detection bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    bbox_gt : a list of 4 elements or a numpy array of 4 elements
        Ground truth bbox.
        Format [top-left-x, top-left-y , bottom-right-x, bottom-right-y].
    weight : a double value, optional
        The weighting parameter of the linear combination of recall and iou metrics.
        The default is 0.5

    Returns
    -------
    fmeasure : numpy scalar of type float
        The weighted harmonic mean of Recall and IoU metrics.
    """
    if (weight <= 0 or weight >= 1):
        raise ValueError("Weight parameter in hybrid metric must be in range (0,1)")

    weight = min(max(weight, 0), 1)

    recall = calculate_recall(bbox_det, bbox_gt)
    iou = calculate_iou(bbox_det, bbox_gt)

    #return 1/(weight * 1/recall + (1 - weight) * 1/iou)
    return (weight**2 + 1)/(weight * 1/recall + 1/iou)

def metrics_gen(mdict, func):
    """
    A utility generator function that yields a pair of matching indices
    from the set of detection and ground truth bboxes.
    In case of an unmatched gt bbox it yields a (None, gt_bbox_index) tuple
    whereas if the unmatched bbox belongs to the detection set it yields 
    a (det_bbox_index, None) tuple.

    Parameters
    ----------
    mdict : dictionary
         A dictionary of det_bbox, gt_bbox info, their matching pairs list
         and the image filename.
    func : function
        A function that calculates a metric given two bboxes.
        One of: calculate_iou, calculate_recall, calculate_precision
                calculate_lpmetric.

    Yields
    ------
    A two element tuple
        A two element tuple of indices. The first is an index to the 
        detection set while the other is an index to the ground truth set.

    """
    for mpair in mdict["matches"]:
        if mpair[0] == None:
            yield Errors.FN.value
        elif mpair[1] == None:
            yield Errors.FP.value
        else:
            yield func( mdict["det"][mpair[0]]["box_points"],
                        mdict["gt"][mpair[1]]["bbox"] )

def calculate_bbox_metrics(mdict_l, weight=None):
    """
    Calculates the bounding box metrics.

    Parameters
    ----------
    mdict_l : a list of dictionaries
        A list of dictionaries that contain the detected bbox coordinates along 
        with their confidence value, the ground truth bbox coordinates, the
        image filename and a list of tupled pair integers representing the indices
        of the matching bboxes from each set.

    Returns
    -------
    metrics : a dictionary of 1d numpy arrays
        A dictionary that contains the 1d numpy arrays of the metrics
        iou, recall, lpmetric and the confidence value of detected bboxes.

    """
    metrics = {"iou": list(),
               "recall": list(),
               "precision": list(),
               "lpmetric1": list(),
               "lpmetric2": list(),
               "confidence": list() }

    for mdict in mdict_l:
        metrics["iou"] += [ m for m in metrics_gen(mdict, calculate_iou) ]
        metrics["recall"] += [ m for m in metrics_gen(mdict, calculate_recall) ]
        metrics["precision"] += [m for m in metrics_gen(mdict, calculate_precision) ]
        metrics["lpmetric1"] += [m for m in metrics_gen(mdict, calculate_lpmetric1) ]
        metrics["lpmetric2"] += [m for m in metrics_gen(mdict, calculate_lpmetric2) ]
        metrics["confidence"] += [ det["confidence"] for det in mdict["det"] if det]

    if weight is not None:
        metrics["hybrid"] = list()
        metrics["fmeasure"] = list()
        metrics["lpcomb"] = list()

        lambda_hybrid =\
            lambda bb_det, bb_gt: calculate_hybrid(bb_det, bb_gt, weight)
        lambda_lp_comb =\
            lambda bb_det, bb_gt: calculate_lp_comb(bb_det, bb_gt, weight)
        lambda_fmeasure =\
            lambda bb_det, bb_gt: calculate_fmeasure(bb_det, bb_gt, weight)

        for mdict in mdict_l:
            metrics["hybrid"] += [ m for m in metrics_gen(mdict, lambda_hybrid) ]
            metrics["lpcomb"] +=\
                [ m for m in metrics_gen(mdict, lambda_lp_comb) ]
            metrics["fmeasure"] +=\
                [ m for m in metrics_gen(mdict, lambda_fmeasure) ]
    return metrics

def calculate_pr(pr_metric, n_gt):
    """
    Precision recall curve calculation.

    Parameters
    ----------
    pr_metric : A dictionary of tupled precision - recall numpy 
        A dictionary of tupled precision - recall 1d numpy arrays.
    n_gt : unsigned int
        The cardinality of the ground truth bboxes set.

    Returns
    -------
    dict: a dictionary of 1d numpy arrays
         A dictionary of precision - recall 1d numpy arrays.

    """
    tp, fp = (0, 0)

    recall = np.empty( len(pr_metric), dtype='double')
    precision = np.empty( len(pr_metric), dtype='double')

    for i in range(len(pr_metric)):
        tp += round(pr_metric[i])
        fp += 1 - round(pr_metric[i])
        recall[i] = tp / n_gt
        precision[i] = tp / (tp + fp)
    return dict({"precision": precision, "recall": recall})

def calculate_performance_metrics(metrics, thres=0):
    """
    Calculate the precision recall curves of the classifier for each metric.

    Parameters
    ----------
    metrics : a dictionary of 1d numpy arrays
        The dictionary contains the 1d arrays of each metric.
    thres : int, optional
        A threshold value between 0 and 1. The default is 0.

    Returns
    -------
    pr_curves : a dict of dicts
        A dictionary of pr curves for each metric, e.g. :
        Key          Values
        ---          ------
        "iou"        {"precision": 1d np array, "recall": 1d np array}
        "recall"     {"precision": 1d np array, "recall": 1d np array}
        "precision"  {"precision": 1d np array, "recall": 1d np array}
        "lpmetric"   {"precision": 1d np array, "recall": 1d np array}
        "hybrid"     {"precision": 1d np array, "recall": 1d np array}
    """
    def threshold_array(arr, thres=0):
        arr = np.asarray(arr) # ensure numpy array
        arr[arr>=thres], arr[arr<thres] = 1, 0

    pr_curves = dict()
    inds = np.argsort(metrics["confidence"])

    for metric_key in [key for key in metrics.keys() if key != 'confidence']:
        metric_arr = metrics[metric_key]

        tmp = np.asarray(metric_arr)
        metric_arr_sorted = tmp[tmp!=Errors.FN.value][ inds[::-1] ] # remove FN

        # ground truth objects (total bboxes after FN removal - flase detection bboxes)
        n_gt_bboxes =\
            len(metric_arr) - len(metric_arr_sorted[metric_arr_sorted==Errors.FP.value])

        threshold_array(metric_arr_sorted, thres)
        pr_curves[metric_key] = calculate_pr(metric_arr_sorted, n_gt_bboxes)

    return pr_curves

def pareto_front_pr_curve(pr_curves):
    pr_curves_paretto = copy.deepcopy(pr_curves)
    ap_scores = dict()

    for metric_key, pr_arr in pr_curves_paretto.items():
        precision_rev = pr_arr['precision'][::-1]
        recall_rev = pr_arr["recall"][::-1]

        curr_max = precision_rev[0]
        curr_max_idx, area = (0, 0)

        while (curr_max_idx < len(precision_rev)):
            next_max_idx =\
                np.argmax(precision_rev[curr_max_idx: ] > curr_max) + curr_max_idx

            next_max = precision_rev[next_max_idx]

            precision_rev[curr_max_idx : next_max_idx] = curr_max

            if curr_max_idx == next_max_idx:
                break

            area += curr_max * (recall_rev[curr_max_idx] - recall_rev[next_max_idx])
            curr_max_idx = next_max_idx
            curr_max = next_max

        area += curr_max * (recall_rev[next_max_idx] - recall_rev[-1])

        ap_scores[metric_key] = area
    return pr_curves_paretto, ap_scores

def front_interp_pr_curve(pr_curves):
    """
    Max front line interpolation of the precision recall curve.
    Due to the nature of the precision - recal curve, the max front is
    calculated in reverse, starting from the last points in the arrays.
    
    

    Parameters
    ----------
    pr_curves : a dict of dicts that contain the PR numpy arrays
        A dictionary of metrics that contain a dictionary of their 
        respective precision and recall arrays.

    Returns
    -------
    pr_curves_interp : a dict of dicts that contain the interpolated PR arrays

    ap_scores : a dict of float values
        Average precision scores of each metric

    """
    pr_curves_interp = copy.deepcopy(pr_curves)
    ap_scores = dict()

    for metric_key, pr_arr in pr_curves_interp.items():

        # reverse presicion and recall arrays
        precision_rev = pr_arr['precision'][::-1]
        recall_rev = pr_arr["recall"][::-1]

        # initialization
        curr_max = precision_rev[0]
        curr_max_idx, area, dp, dr = 2*(0, 0)

        while (curr_max_idx < len(precision_rev)):
            next_max_idx =\
                np.argmax(precision_rev[curr_max_idx: ] > curr_max) + curr_max_idx

            # a sequence of equal recall values leads to a vertical line in the curve
            # find the index of the first recall value that is less than the repeating
            vert_line_len = np.argmax(recall_rev[next_max_idx: ] < recall_rev[next_max_idx])

            # move to the repeating value's last index in the sequence
            # vert_line_len can be 0 if all the residual values are larger than the repeating
            # so take the maximum of the repeating value multiplicity and 0
            next_max_idx += max(vert_line_len - 1, 0)

            next_max = precision_rev[next_max_idx] # next max precision value

            # avoid  area and line points calculation in case of a vertical line
            if (recall_rev[next_max_idx] != recall_rev[curr_max_idx]):
                # differential value of recall
                dr = recall_rev[curr_max_idx] - recall_rev[next_max_idx]
                # differential value of precision
                dp = curr_max - next_max

                # inclination
                incl = dp / dr

                # interval interpolation with a line

                # slice the recall array and use it as the linear space of the
                # interpolation (r_lin - r[initial])
                recall_lin =\
                    recall_rev[curr_max_idx : next_max_idx] - recall_rev[curr_max_idx]

                # line interpolation
                precision_rev[curr_max_idx : next_max_idx] =\
                    incl * recall_lin + curr_max

                # area of the interval (triangle + rectangle)
                area += (np.abs(dp)/2 + curr_max) * dr

            # break when the curve maximum is reached
            # when there are no more max-valued indices displacement is 0
            if curr_max_idx == next_max_idx:
                break

            curr_max_idx = next_max_idx
            curr_max = next_max

        # area of the last segment
        dr = recall_rev[next_max_idx] - recall_rev[-1]
        dp = precision_rev[-1] - curr_max
        area += (dp/2 + curr_max) * dr

        # average precision score
        ap_scores[metric_key] = area
    return pr_curves_interp, ap_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precision - Recall metrics')
    parser.add_argument('--size', '-s',
                        type=float,
                        default=12,
                        help="the size (non-metric) of the output png file",
                        required=False)
    parser.add_argument('--dpi','-d',
                        type=int,
                        default=200,
                        help="sets the dpi scale of the output png file",
                        required=False)
    parser.add_argument('--thres','-t',
                        type=float,
                        default=0.75,
                        help="the threshold value",
                        required=False)
    parser.add_argument('--weight', '-w',
                        type=float,
                        default=None,
                        help="weight coefficient of the hybrid metrics",
                        required=False)
    parser.add_argument('--dirpath', '-p',
                        type=str,
                        default=os.getcwd(),
                        help="full path of the plots directory",
                        required=False)
    parser.add_argument('--pkl',
                        type=str,
                        default='mdict_list.pkl',
                        help="a pickle file of a mdict_list",
                        required=False)
    parser.add_argument('--plot',
                        action='store_true',
                        help="show plot switch",
                        required=False)
    parser.add_argument('--ratio',
                        type=float,
                        default=1.2,
                        help="plot png ascpect ratio",
                        required=False)
    args = parser.parse_args()

    size = args.size
    dpi = args.dpi
    thres = args.thres
    weight = args.weight
    dirpath = args.dirpath
    pickle_file = args.pkl
    plot_flag = args.plot
    ratio = args.ratio

    mdict_l = pickle.load(open(pickle_file, 'rb'))
    metrics = calculate_bbox_metrics(mdict_l, weight)

    # calculate precision - recall arrays
    pr_curves = calculate_performance_metrics(metrics, thres)

    # take the max front of the curves
    pr_curves_pareto, ap_pareto = pareto_front_pr_curve(pr_curves)

    # interpolate max front points
    pr_curves_interp, ap_interp = front_interp_pr_curve(pr_curves)

    # omit confidence from the list of metric keys
    metric_keys = [key for key in metrics.keys() if key not in ('confidence')]

    # plot cell layout
    ncols = int(np.rint(np.sqrt(len(metric_keys))))
    nrows = int(np.ceil(len(metric_keys)/ncols))

    # figure and grid initialization
    fig = pyplot.figure(figsize = (ratio*size, size))
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    for metric in metric_keys:
        i = metric_keys.index(metric)//ncols
        j = metric_keys.index(metric) - ncols * i

        j = slice(j,None)\
            if len(metric_keys) == metric_keys.index(metric)+1 else j
        ax = fig.add_subplot(spec[i, j])

        ax.plot(pr_curves[metric]["recall"], pr_curves[metric]["precision"])
        ax.plot(pr_curves_pareto[metric]["recall"],
                pr_curves_pareto[metric]["precision"],
                label = '%s AP=%.2f %%' % ('Pareto front', 100*ap_pareto[metric]))
        ax.plot(pr_curves_interp[metric]["recall"], 
                pr_curves_interp[metric]["precision"],
                linestyle = '--',
                label = '%s AP=%.2f %%' % ('Front interpolation', 100*ap_interp[metric]))

        ax.set_xlim(0, pr_curves[metric]["recall"][-1] + 1e-2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        ax.legend(shadow=False, loc='lower left')
        if weight is not None and metric == 'hybrid':
            formula = "\n $%s= %.2f \cdot Recall + %.2f \cdot Iou $"\
                % (metric.capitalize(), weight, 1-weight)
            ax.set_title("Precision x Recall curve \n%s, mAP=%.2f%%, $%s_{th}=%.2f$" % 
                         (r'Vehicle', 100*ap_pareto[metric],
                          metric.capitalize(), thres) + formula)
        elif weight is not None and metric == 'fmeasure':
            formula ="\n$%s = \dfrac{%.2f\cdot Recall*Iou}{%.2f\cdot Iou+Recall}$"\
                % (metric.capitalize(), weight**2 + 1, weight**2)
            ax.set_title("Precision x Recall curve \n%s, mAP=%.2f%%, $%s_{th}=%.2f$"\
                         % (r'Vehicle', 100*ap_pareto[metric],
                            metric.capitalize(), thres) + formula)
        elif weight is not None and metric == 'lpcomb':
            formula = "\n $%s = %.2f \cdot Lpmetric2+ %.2f \cdot Precision $"\
                % (metric.capitalize(), weight, 1-weight)
            ax.set_title("Precision x Recall curve \n%s, mAP=%.2f%%, $%s_{th}=%.2f$"\
                         % (r'Vehicle', 100*ap_pareto[metric],
                          metric.capitalize(), thres) + formula)
        else:
            ax.set_title('Precision x Recall curve \n%s, mAP=%.2f%%, $%s_{th}=%.2f$'\
                         % (r'Vehicle', 100*ap_pareto[metric],
                            metric.capitalize(), thres))
        ax.grid()

    if plot_flag:
        pyplot.show()
    fig.tight_layout()

    try:
        if os.path.isfile(dirpath):
            raise ValueError
    except ValueError:
        print('Argument --dirpath must be a directory not a file')
        print('Saving to default execution directory')
        dirpath = os.getcwd()

    if not os.path.isdir( os.path.abspath(dirpath) ):
        os.mkdir( os.path.normpath( os.path.abspath(dirpath) ) )

    if weight is not None:
        fig.savefig(os.path.join(os.path.abspath(dirpath),
                    'pr_t%d_w%d' % (thres*100, weight*100) +'.png'),
                    dpi = dpi)
    else:
        fig.savefig(os.path.join(os.path.abspath(dirpath),
                    'pr_t%d' % (thres*100) +'.png'),
                    dpi = dpi)
