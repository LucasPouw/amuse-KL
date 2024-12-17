import cv2
import os
import sys
import glob
import argparse


def moviemaker(image_folder, video_name, fps=10, height=1080, width=1440):

    if not os.path.isdir(image_folder):
        sys.exit(f'{image_folder} is not a folder. Quitting.')

    print(f'Generating video {video_name} from {image_folder} with {fps} fps.')

    unsorted_images = glob.glob(image_folder+'/' + '*.png')
    image_numbers = [image_name.split('-')[-1].split('.')[0] for image_name in unsorted_images]
    image_numbers = list(map(int, image_numbers))
    _, sorted_images = zip(*sorted(zip(image_numbers, unsorted_images)))

    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    for image_name in sorted_images:
        img = cv2.imread(image_name)
        img = cv2.resize(img, (width, height))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a video from a folder of images.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to folder containing images. Required.')
    parser.add_argument('--video_name', type=str, required=True, help='Name of the video to be generated. Required.')
    parser.add_argument('--fps', default=10, type=int, help='Frames per second of the video.')
    parser.add_argument('--height', default=1080, type=int, help='Height of the video.')
    parser.add_argument('--width', default=1440, type=int, help='Width of the video.')
    args = parser.parse_args()

    video_name = args.video_name
    image_folder = args.image_folder
    fps = args.fps
    height = args.height
    width = args.width

    moviemaker(image_folder, video_name, fps, height, width)
