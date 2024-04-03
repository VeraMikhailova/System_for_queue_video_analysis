import os.path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
import numpy as np
import torch
import cv2

SETTINGS_PATH = "./settings.txt"


# Prints little info about this script
def print_help(output_default_path, output_quality, seconds_in_queue):
    print("------------ENG---------------")
    print(
        f"This is a script for analysing queues.\nEach camera can have different settings file. You can select not existent file to create new one.\nControls: ESC,q - exit. Enter - apply. u - undo. Double click - select point (if new settings file is used you have to provide queue's polygon by selecting points).\nOutput video file will be written to: {output_default_path}. Opportunity to move it will become available after processing the video.\nCurrent output quality is set to {output_quality}.\nPerson will be counted as one in queue if and only if he is inside the polygon set for this camera for at least {seconds_in_queue} seconds.\nThis constants can be changed in {SETTINGS_PATH}"
    )
    print("------------RUS---------------")
    print(
        f"Это скрипт для проведения анализа очереди.\nКаждая камера может иметь свой файл настроек. Вы можете выбрать несуществующий файл, чтобы создать новый.\nУправление: ESC,q - выход. Enter - применить. u - отменить. Двойной клик - выбрать точку (если создаётся новый файл настроек, то необходимо ограничить ожидаемую зону очереди многоугольником, для чего выбрать его вершины).\nВидео после обработки будет записано в файл: {output_default_path}. Возможность переместить его будет предоставлена после обработки видео.\nТекущее качество видео: {output_quality}.\nЧеловек будет считаться стоящим в очереди, если и только если он находится в многоугольнике, заданном для текущей камеры не менее {seconds_in_queue} секунд.\nЭти константы можно поментять в файле {SETTINGS_PATH}"
    )
    print("------------------------------")


# Reads nodes for queue bounding from file
def get_nodes(path):
    # Exclude not files and almost empty files (>=1 for n, >=3 for each line, >=4 eol's, means >=14)
    if (not os.path.isfile(path)) or os.stat(path).st_size < 14:
        return [None, None]
    nodes_queue = []
    nodes_staff = []
    with open(path, "rt") as file:
        n = int(file.readline().strip())
        # Polygon must consist of at least 3 nodes
        if n < 3:
            return [None, None]
        for i in range(n):
            nodes_queue.append(list(map(int, file.readline().strip().split())))
        # And for staff:
        line = file.readline().strip()
        if len(line) == 0:
            return [nodes_queue, None]
        n = int(line)
        if n < 3:
            return [nodes_queue, None]
        for i in range(n):
            nodes_staff.append(list(map(int, file.readline().strip().split())))
    return [nodes_queue, nodes_staff]


# Saves nodes for queue bounding to file
def save_nodes(path, nodes):
    if not nodes[0] is None:
        with open(path, "wt") as file:
            # Save nodes for queue
            file.write(str(len(nodes[0])))
            file.write("\n")
            for node in nodes[0]:
                file.write(str(node[0]))
                file.write(" ")
                file.write(str(node[1]))
                file.write("\n")
            # Save nodes for staff
            if not nodes[1] is None:
                file.write(str(len(nodes[1])))
                file.write("\n")
                for node in nodes[1]:
                    file.write(str(node[0]))
                    file.write(" ")
                    file.write(str(node[1]))
                    file.write("\n")


def analyse(capture, nodes_queue, nodes_staff, out_video, seconds_in_queue):
    # Load YOLO model using m for speed, specialization - None, verbose - False
    model = YOLO("yolov8m.pt", None, False)
    # Fuse Conv2d and BatchNorm2d layers in the model to improve inference speed.
    model.fuse()
    # Use cpu or gpu for analysis
    FORCE_CPU = True
    model.to("cpu" if (FORCE_CPU or not torch.cuda.is_available()) else "cuda")

    # Function for generation polygon and np points array from nodes
    def nodes2poly(nodes):
        pts = np.array(nodes, np.int32)
        pts = pts.reshape((-1, 1, 2))
        return [Polygon(nodes), pts]

    # Start without polygon
    polygon_queue = None
    pts_queue = None
    polygon_staff = None
    pts_staff = None
    # Set time before person counts as queued
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("Using seconds_in_queue delay:", seconds_in_queue)
    print("FPS:", fps)
    delta_t = 1 / fps
    current_time = 0
    persons_enter_time = {}
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # We can use native YOLO's tracking ability without external tools
        result = model.track(frame, persist=1)[0]
        # We will draw image in custom way
        image = np.copy(result.orig_img)
        # Create window
        WAITKEY_TIME = 5
        window_name = "Video analysis"
        cv2.namedWindow(window_name)

        # If polygon and points np array are not set then we need to read this objects from user
        def get_nodes_from_user(object_name, orig_image):
            image = np.copy(orig_image)
            cv2.putText(
                image,
                object_name,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            nodes = []

            def callback(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDBLCLK:
                    p = [x, y]
                    nodes.append(p)
                    cv2.circle(image, p, 5, (255, 255, 255), -1)

            cv2.setMouseCallback(window_name, callback)
            while True:
                cv2.imshow(window_name, image)
                key = cv2.waitKey(WAITKEY_TIME)
                if key == 13:
                    if len(nodes) >= 3:
                        break
                if key & 0xFF == ord("q") or key == 27:
                    cv2.destroyAllWindows()
                    return None
                if key & 0xFF == ord("u") and len(nodes) != 0:
                    nodes.pop()
                    image = np.copy(orig_image)
                    cv2.putText(
                        image,
                        object_name,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    for p in nodes:
                        cv2.circle(image, p, 5, (255, 255, 255), -1)
            cv2.setMouseCallback(window_name, lambda *args: None)
            return nodes

        if pts_queue is None:
            if nodes_queue is None:
                nodes_queue = get_nodes_from_user("Queue:", result.orig_img)
                if nodes_queue is None:
                    return [nodes_queue, nodes_staff]
            polygon_queue, pts_queue = nodes2poly(nodes_queue)
        if pts_staff is None:
            if nodes_staff is None:
                nodes_staff = get_nodes_from_user("Staff:", result.orig_img)
                if nodes_staff is None:
                    return [nodes_queue, nodes_staff]
            polygon_staff, pts_staff = nodes2poly(nodes_staff)
        # Draw polygons
        cv2.polylines(image, [pts_queue], 1, (0, 0, 255), 2)
        cv2.polylines(image, [pts_staff], 1, (255, 0, 0), 2)
        # Collect results from YOLO
        result.boxes.cpu()
        n = len(result.boxes.cls)
        where = []
        for i in range(n):
            if result.boxes.cls[i] == 0:
                p = [
                    int(result.boxes.xywh[i][0]),
                    int(result.boxes.xywh[i][1] + result.boxes.xywh[i][3] / 2),
                ]
                # Draw results on this image: person rect and bottom point
                cv2.rectangle(
                    image,
                    (int(result.boxes.xyxy[i][0]), int(result.boxes.xyxy[i][1])),
                    (int(result.boxes.xyxy[i][2]), int(result.boxes.xyxy[i][3])),
                    (0, 255, 0),
                    2,
                )
                cv2.circle(image, p, 10, (255, 0, 0), -1)
                p.append(int(result.boxes.id[i]))
                where.append(p)
        # Counting number of persons in queue, delayed and outside of it
        counter_in = 0
        counter_in_delayed = 0
        counter_out = 0
        counter_staff = 0
        counter_warning = 0
        for p in where:
            person_id = p[2]
            p.pop()
            point = Point(p)
            if polygon_staff.contains(point):
                counter_staff += 1
                if polygon_queue.contains(point):
                    counter_warning += 1
                # In case of some staff member was in queue
                if person_id in persons_enter_time:
                    del persons_enter_time[person_id]
            else:
                if polygon_queue.contains(point):
                    if person_id not in persons_enter_time:
                        persons_enter_time[person_id] = current_time
                    if persons_enter_time[person_id] + seconds_in_queue <= current_time:
                        counter_in += 1
                    else:
                        counter_in_delayed += 1
                else:
                    counter_out += 1
                    if person_id in persons_enter_time:
                        del persons_enter_time[person_id]
        # Print results to stdout and image
        print(
            "Queue:",
            counter_in,
            ". Delayed:",
            counter_in_delayed,
            ". Staff:",
            counter_staff,
            ". Others:",
            counter_out,
            ".",
        )
        print("Current time from video start:", current_time)
        # Alert special cases:
        warned = False
        if counter_in != 0 and counter_staff == 0:
            print("Warning: no staff members while not empty queue!")
            warned = True
        if counter_warning != 0:
            print(
                f"Warning: there are {counter_warning} persons in both polygons! Your camera looks misconfigured!"
            )
            warned = True
        cv2.putText(
            image,
            str(counter_in) + ":" + str(counter_in_delayed) + (" !" if warned else ""),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Show image and exit at 'q' key press
        cv2.imshow(window_name, image)
        out_video.write(image)
        key = cv2.waitKey(WAITKEY_TIME)
        if key & 0xFF == ord("q") or key == 27:
            break
        current_time += delta_t
    cv2.destroyAllWindows()
    return [nodes_queue, nodes_staff]


# Default settings
output_quality = 100
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
output_default_path = "./output.avi"
seconds_in_queue = 3
# If settings file does not exist or is almost empty (>=1 quality; >=4 format; >=1 seconds; >=1 filename; >=4 eol) then create it else read current settings
if (not os.path.isfile(SETTINGS_PATH)) or os.stat(SETTINGS_PATH).st_size < 11:
    with open(SETTINGS_PATH, "wt") as file:
        file.write("100\nDIVX\n./output.avi\n3\n")
else:
    with open(SETTINGS_PATH, "rt") as file:
        output_quality = int(file.readline().strip())
        fourcc = cv2.VideoWriter_fourcc(*file.readline().strip())
        output_default_path = file.readline().strip()
        seconds_in_queue = float(file.readline().strip())
# Infrom user about current settings and some helpful information about this script
print_help(output_default_path, output_quality, seconds_in_queue)
# Read paths from user
video_path = input("Provide video path for analysis: ").strip()
nodes_path = input("Provide camera's settings path: ").strip()
# Create video capture and read properties
capture = cv2.VideoCapture(video_path)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))
# Create video writer and set quality
out_video = cv2.VideoWriter(output_default_path, fourcc, fps, (width, height))
out_video.set(cv2.VIDEOWRITER_PROP_QUALITY, output_quality)
# Read camera settings (polygon)
nodes = get_nodes(nodes_path)
# Perform analysis
nodes = analyse(capture, nodes[0], nodes[1], out_video, seconds_in_queue)
# Save camera settings (polygon)
save_nodes(nodes_path, nodes)
# Release resources
out_video.release()
capture.release()
# If no correct detection was performed:
if nodes[0] is None or nodes[1] is None:
    print(
        "No correct detection was performed. Create correct camera's settings file using this script."
    )
    exit(1)
# Opportunity to move resulting file
out_path = input(
    f"Where do you whant to place analysed video (path or NONE; current is {output_default_path}): "
).strip()
if out_path != "NONE":
    os.rename(output_default_path, out_path)
