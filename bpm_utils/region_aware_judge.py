# Region-Aware Judge

def determine_size(bbox1, bbox2, entry, threshold1=0.8, threshold2=1.2):
    # bbox1 from edited image, bbox2 from original image
    if len(bbox1) == 0 or len(bbox2) == 0:
        return 1
    
    bbox1 = bbox1[0]
    bbox2 = bbox2[0]
    relation = entry['size']
    if 'change size' in entry['edit_type']:
        # bbox1 from edited image, bbox2 from original image
        before_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        after_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        ratio = after_size / before_size

        if relation == 'bigger':
            return ratio > threshold2
        elif relation == 'smaller':
            return ratio < threshold1
        elif relation == 'unchanged':
            return threshold1 <= ratio <= threshold2
    return 1

def determine_position(bbox1, bbox2, relation, iou_thresh, distance_thresh):
    # bbox1 from edited image, bbox2 from original image

    # Calculate the centers of the two bboxes
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

    # Calculate the IoU
    inter_xmin = max(bbox1[0], bbox2[0])
    inter_ymin = max(bbox1[1], bbox2[1])
    inter_xmax = min(bbox1[2], bbox2[2])
    inter_ymax = min(bbox1[3], bbox2[3])
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    iou = inter_area / (bbox1_area + bbox2_area - inter_area)

    # Judge the position
    if 'right' in relation:
        # x1 > x2
        rule1 = center1[0] > center2[0]
        # x1 - x2 > y1 - y2
        rule2 = abs(center1[0] - center2[0]) > abs(center1[1] - center2[1])
        # iou < thresh
        rule3 = iou < iou_thresh

    elif 'left' in relation:
        # x1 < x2
        rule1 = center1[0] < center2[0]
        # x2 - x1 > y1 - y2
        rule2 = abs(center1[0] - center2[0]) > abs(center1[1] - center2[1])
        # iou < thresh
        rule3 = iou < iou_thresh
    
    elif 'above' in relation:
        # y1 < y2
        rule1 = center1[1] < center2[1]
        # y2 - y1 > x1 - x2
        rule2 = abs(center1[1] - center2[1]) > abs(center1[0] - center2[0])
        # iou < thresh
        rule3 = iou < iou_thresh
    
    elif 'below' in relation:
        # y1 > y2
        rule1 = center1[1] > center2[1]
        # y1 - y2 > x1 - x2
        rule2 = abs(center1[1] - center2[1]) > abs(center1[0] - center2[0])
        # iou < thresh
        rule3 = iou < iou_thresh
    
    elif 'inside' in relation:
        # bbox1 in bbox2
        rule1 = bbox1[0] > bbox2[0] and bbox1[1] > bbox2[1] and bbox1[2] < bbox2[2] and bbox1[3] < bbox2[3]
        rule2, rule3 = True, True

    elif 'unchanged' in relation:
        # d(center1, center2) < thresh
        rule1 = abs(center1[0] - center2[0]) + abs(center1[1] - center2[1]) < distance_thresh
        rule2, rule3 = True, True
    else:
        return True
    
    return rule1 and rule2 and rule3

def determine_position_multi_bbox(bbox1, bbox2, entry, iou_thresh=0.4, distance_thresh=40):
    if len(bbox1) == 0 or len(bbox2) == 0:
        return 1
    # bbox1 from edited image, bbox2 from original image
    relation = entry['position']
    if 'none' in entry['original_part'] and entry['being_added']:
        if len(bbox1) == 1:
            return int(determine_position(bbox1[0], bbox2[0], relation, iou_thresh, distance_thresh))
        else:
            flag1 = all([determine_position(b1, bbox2[0], relation, iou_thresh, distance_thresh) for b1 in bbox1])
            flag2 = any([determine_position(b1, bbox2[0], relation, iou_thresh, distance_thresh) for b1 in bbox1])
            if flag2:
                if flag1:
                    return 1
                else:
                    return 0.5
            else:
                return 0
    return 1