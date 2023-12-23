import cv2

def read_image(img):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to read the image at {img}")
    return image

def read_video(video):
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Error: Unable to open the video at {video}")
    return capture

def is_similar_template(gray_frame, template, treshold_value):
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val >= treshold_value

def main(video, img):
    try:
        image = read_image(img)
        template = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        capture = read_video(video)

        image_count = 0
        treshold_value = 0.9

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if is_similar_template(gray_frame, template, treshold_value):
                image_count += 1
                if image_count % 10 == 0:
                    print(f"Found {image_count} images so far.")

        capture.release()

        print(f"Total frames with the specified image: {image_count}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    video = "output.avi"
    img = "ezhov.png"

    main(video, img)
