#Student Id: 20319961  Student Name: Jiazhu Quan
#This code uses the OpenCV library
#To run this code, please use "python test.py -v <name.format>"
import cv2
import numpy as np
import os.path as path
import argparse

font = cv2.FONT_HERSHEY_SIMPLEX

class Person():
    def __init__(self, identifier, frame, track_window):
        # Initialize a Person object with an identifier, initial track window, and an empty history
        self.identifier = int(identifier)
        self.track_window = track_window
        self.history = []
        x, y, w, h = track_window
        self.center = np.array([x + w / 2, y + h / 2])
        # Draw a rectangle around the person on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Add text showing the person's ID near the rectangle
        cv2.putText(frame, "ID: %d" % self.identifier, (x, y - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    def update(self, frame, new_window):
        # Update the person's track window and history
        x, y, w, h = new_window
        self.center = np.array([x + w / 2, y + h / 2])
        self.history.append(tuple(self.center.astype(int)))
        # Update the rectangle around the person on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Draw a circle at the center of the person
        cv2.circle(frame, tuple(self.center.astype(int)), 2, (0, 255, 0), -1)

def filter_contours(contours, min_area=300, max_area=10000, min_aspect_ratio=0.2, max_aspect_ratio=1.2):
    # Filter contours based on area and aspect ratio
    filtered_contours = []
    for c in contours:
        if max_area > cv2.contourArea(c) > min_area:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                filtered_contours.append((x, y, w, h))
    return filtered_contours

def main(video_file):
    # Main function to process the input video file
    camera = cv2.VideoCapture(video_file)
    bs = cv2.createBackgroundSubtractorKNN()
    people = {}  # Dictionary to store person objects
    firstFrame = True  # Flag to indicate the first frame
    frames = 0  # Counter for frames processed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for writing video output
    out = cv2.VideoWriter('output.avi', fourcc, 25, (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    prev_people = {}  # Dictionary to store previous positions of people

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break
        fgmask = bs.apply(frame)
        if frames < 20:
            frames += 1
            continue
        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = filter_contours(contours)

        counter = 0
        for (x, y, w, h) in filtered_contours:
            # Process each filtered contour
            if firstFrame:
                # If it's the first frame, create a new Person object
                people[counter] = Person(counter, frame, (x, y, w, h))
            else:
                # Update existing person or create a new one
                if counter in people:
                    old_window = people[counter].track_window
                    people[counter].update(frame, (x, y, w, h))
                    new_window = people[counter].track_window
                    dx = new_window[0] - old_window[0]
                    dy = new_window[1] - old_window[1]
                    for i in range(len(people[counter].history)):
                        people[counter].history[i] = (people[counter].history[i][0] + dx,
                                                      people[counter].history[i][1] + dy)
                else:
                    people[counter] = Person(counter, frame, (x, y, w, h))
            counter += 1

        # Update previous positions of people
        for person in people.values():
            prev_people[person.identifier] = person.center

        # Match current contours with previous positions to update or create people
        for (x, y, w, h) in filtered_contours:
            for person_id, prev_center in prev_people.items():
                dist = np.linalg.norm(np.array([x + w / 2, y + h / 2]) - prev_center)
                if dist < 5:
                    if person_id in people:
                        old_window = people[person_id].track_window
                        people[person_id].update(frame, (x, y, w, h))
                        new_window = people[person_id].track_window
                        dx = new_window[0] - old_window[0]
                        dy = new_window[1] - old_window[1]
                        for i in range(len(people[person_id].history)):
                            people[person_id].history[i] = (people[person_id].history[i][0] + dx,
                                                             people[person_id].history[i][1] + dy)
                    else:
                        people[person_id] = Person(person_id, frame, (x, y, w, h))

        firstFrame = False
        frames += 1
        # Draw the history of each person on the frame
        for person in people.values():
            for point in person.history:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
        # Show the frame with annotations
        cv2.imshow("Person tracking", frame)
        # Write frame to output video
        out.write(frame)
        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(110) & 0xff == 27:
            break
    out.release()
    camera.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="Path to input video file")
    args = parser.parse_args()
    if args.video:
        # Call main function with input video file path
        main(args.video)
    else:
        print("Please provide the path to the input video file using -v or --video option.")
