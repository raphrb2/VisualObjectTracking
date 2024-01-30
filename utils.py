import cv2

def update_tracks(max_id, assignments, nb_obj):
    assignment_dict = {col: id for col, id in assignments}
    updated_tracks = [assignment_dict.get(i, max_id+i+1) for i in range(nb_obj)]
    if len(updated_tracks) > 0:
        max_id = max(max_id, max(updated_tracks))
    return updated_tracks


def display_boxes(boxes, labels, tracks):
    image_path = "ADL-Rundle-6/img1/" + str(int(labels)).zfill(6) + ".jpg"
    image = cv2.imread(image_path)

    for box, track in zip(boxes, tracks):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255), 2)
        cv2.putText(image, str(track), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", image)

def frame_to_boxes(frame):
    return frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values

def box2centroid(box):
    return box[0] + (box[2] / 2), box[1] + (box[3] / 2)

def centroid2box(centroid, width, height):
    x = centroid[0] - (width / 2)
    y = centroid[1] - (height / 2)
    return x, y, width, height

def box_display_2(boxes, labels, tracks):
    image_path = "ADL-Rundle-6/img1/" + str(int(labels)).zfill(6) + ".jpg"
    try:
        image = cv2.imread(image_path)
        for box, track in zip(boxes, tracks):
            x, y, width, height = box
            cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 2)
            cv2.putText(image, str(track), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Tracking", image)
        cv2.waitKey(10)
    except Exception as e:
        print(f"Error displaying image: {e}")