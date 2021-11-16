"""
need torch 1.7 & python 3.8 and up versions.
To install dependencies:
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
"""
import pandas
import torch
import cv2
import os
import json
import tensorflow as tf

fps = 30  # frames per second, normal video = 30, so I'm good :)
Green1 = (170, 250, 50)
Green2 = (120, 250, 90)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = 2  # car recognition only, class number at coco dataset...
model.conf = 0.73  # confidence threshold

# Vid handling:
vidcap = cv2.VideoCapture('C:/Users/Israel/PycharmProjects/bootcampYoloEntryProject/Cars_vid2.mp4')

#read frame
success, image = vidcap.read()
img_size_h, img_size_w, rgb = image.shape
size = (img_size_w, img_size_h)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(os.path.join('./vid_conct', 'cars_simple_check2.avi'), fourcc, 30, size, True)
output = [] #data for json
global_counter = 0

with open('json_data_every_car_indexed.json', 'w') as outfile:
    count = 0
    model_output = model(image, size=1024)
    # Draw over image + json data:
    if model_output.pred[0].nelement() != 0:  # tensor isn't empty
        the_tf_shape = tf.shape(model_output.pred[0].cpu())
        num_of_cars = the_tf_shape[0].numpy()
        for i in range(num_of_cars):
            # upper left xy point start from (0,0), so:
            x_left = int(model_output.pred[0][i][0].cpu().numpy().item())
            y_up = int(model_output.pred[0][i][1].cpu().numpy().item())
            x_right = int((model_output.pred[0][i][2].cpu().numpy()).item())
            y_bot = int((model_output.pred[0][i][3].cpu().numpy()).item())
            pred_prob = (model_output.pred[0][i][4].cpu().numpy()).item()
            # class_type = (model_output.pred[0][i][5].cpu().numpy()).item()
            class_type = "car"  # By definition above...
            if pred_prob >= 0.73:  # Doesn't really needed by definition threshold above...
                """#as I know it would find only cars because of model.clases = 2, 
                if class_type == 2:
                    class_type = "car"
                """
                # Draw the bounding box rectangle and label on the image
                cv2.rectangle(image, (x_left, y_up), (x_right, y_bot), Green1, 2)
                text = "{}: {:4f}".format(class_type, pred_prob)
                cv2.putText(image, text, (x_left, y_up - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Green2, 2)

                # json:
                global_counter += 1
                boundingbox = "x_left " + str(x_left) + ' y_up ' + str(y_up) + ' x_right ' + str(x_right) + ' y_bot ' + str(y_bot)
                data = {'Object %d Bounding box' % global_counter: [boundingbox]}

                # Convert the dictionary into DataFrame
                df = pandas.DataFrame(data)

                output.append(df.to_json(orient="records"))
    out.write(image)

    while success:  # get vid frames
        success, image = vidcap.read()
        if not success:  # <=> success == False
            break
        model_output = model(image, size=1024)

        if model_output.pred[0].nelement() != 0:  # check tensor isn't empty
            the_tf_shape = tf.shape(model_output.pred[0].cpu())
            num_of_cars = the_tf_shape[0].numpy()
            for i in range(num_of_cars):

                # upper left xy point start from (0,0), so:
                x_left = int(model_output.pred[0][i][0].cpu().numpy().item())
                y_up = int(model_output.pred[0][i][1].cpu().numpy().item())
                x_right = int((model_output.pred[0][i][2].cpu().numpy()).item())
                y_bot = int((model_output.pred[0][i][3].cpu().numpy()).item())
                pred_prob = (model_output.pred[0][i][4].cpu().numpy()).item()

                # class_type = (model_output.pred[0][i][5].cpu().numpy()).item()
                class_type = "car"
                if pred_prob >= 0.73:
                    # Draw the bounding box rectangle and label on the image
                    cv2.rectangle(image, (x_left, y_up), (x_right, y_bot), Green1, 2)
                    text = "{}: {:4f}".format(class_type, pred_prob)
                    cv2.putText(image, text, (x_left, y_up - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Green2, 2)

                    # json:
                    global_counter += 1
                    boundingbox = "x_left " + str(x_left) + ' y_up ' + str(y_up) + ' x_right ' + str(x_right) + ' y_bot ' + str(y_bot)
                    data = {'Object %d Bounding box' % global_counter: [boundingbox]}

                    # Convert the dictionary into DataFrame
                    df = pandas.DataFrame(data)

                    output.append(df.to_json(orient="records"))
        out.write(image)
    json.dump(output, outfile)
out.release()