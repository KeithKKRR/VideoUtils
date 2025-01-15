import cv2
import dlib
import numpy as np
from skimage import transform as trans

# (dlib.fhog_object_detector)

def get_key_points(image, face, predictor):
    """
    Detect the facial landmarks for the selected face.
    Args:
        image: (np.ndarray in [H, W, 3]) image with faces in RGB order
        face: (dlib_pybind11.rectangle) face rectangle position
        predictor: (dlib.shape_predictor) face key point predictor
    Returns:
        pts: (np.ndarray in [5, 2]) 5 key points position
    """
    shape = predictor(image, face)

    # select the key points for the eyes, nose, and mouth
    left_eye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    right_eye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    left_mouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    right_mouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    pts = np.concatenate([left_eye, right_eye, nose, left_mouth, right_mouth], axis=0)

    return pts


def image_align_and_crop(image, landmarks, output_size, scale=1.25):
    """
    Align face on image, crop and resize to 'output_size'.
    Args:
        image: (np.ndarray in shape [H, W, 3]) im age tensor in RGB order
        landmarks: (np.ndarray in shape [5, 2]) key point landmarks
        output_size: (float, float) output size (height, width)
        scale: (float, default=1.3) scale factor to expand cropped area
    Returns:
        image: (np.ndarray in shape [*output_size, 3]) aligned and cropped face image
        M: (np.ndarray in shape [2, 3]) affine matrix
    """

    template_size = [112, 112]
    dst = np.array([[30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041]], dtype=np.float32)

    # to make sure more centralize facial part
    dst[:, 0] += 8.0

    # add margin
    dst[:, 0] += template_size[0] * (scale - 1) / 2
    dst[:, 1] += template_size[1] * (scale - 1) / 2

    # resize to output size
    dst[:, 0] *= output_size[0] / (template_size[0] * scale)
    dst[:, 1] *= output_size[1] / (template_size[1] * scale)

    # align the face
    landmarks = landmarks.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    M = tform.params[0:2, :]

    # crop the face and mask, if any, and resize to output_size
    image = cv2.warpAffine(image, M, (output_size[1], output_size[0]))

    return image, M


def mask_align_and_crop(mask, M, output_size):
    """
    Crop and resize mask to output_size using affine matrix M.
    Args:
        mask: (np.ndarray in shape [H, W]) mask with value in [0, 1]
        M: (np.ndarray in shape [2, 3]) affine matrix
        output_size: (float, float) output size (height, width)
    Returns:
        mask: (np.ndarray in shape [*output_size] or None) corresponding cropped mask if mask is provided
    """
    mask = cv2.warpAffine(mask, M, (output_size[1], output_size[0]))

    return mask


def paste_face_back_to_image(original_image, cropped_image, M):
    """
    Paste cropped face back to original image.
    Args:
        original_image: (np.ndarray in [H, W, 3]) original image
        cropped_image: (np.ndarray in [*output_size, 3] cropped image
        M: (np.ndarray in shape [2, 3]) affine matrix
    Returns:
        result_image: (np.ndarray in [H, W, 3]) image after pasting cropped face back to original image
    """
    offset = 3
    max_h, max_w = cropped_image.shape[0] - 1, cropped_image.shape[1] - 1
    corners = np.array([[offset, offset],
                        [max_w - offset, offset],
                        [max_w - offset, max_h - offset],
                        [offset, max_h - offset]], dtype=np.float32)
    transformed_corners = cv2.transform(np.array([corners]), cv2.invertAffineTransform(M))[0]

    mask = np.zeros(original_image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, transformed_corners.astype(np.int32), (1, 1, 1))

    output_h, output_w = original_image.shape[:2]
    cropped_image = cv2.warpAffine(cropped_image, cv2.invertAffineTransform(M), (output_w, output_h))

    result_image = mask * cropped_image + (1 - mask) * original_image

    return result_image


def get_first_face_frame_from_video(video_path, face_detector):
    """
    Get first frame with face detected from a video
    Args:
        video_path: (str) video path
        face_detector: dlib face detector
    Returns:
        frame: (np.ndarray) frame ndarray in RGB order, if detected, otherwise, return None
        largest_face: (dlib_pybind11.rectangle) detected largest face
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open video: {video_path}')
        cap.release()
        return None, None

    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            print(f'Video loading error: {video_path}')
            cap.release()
            return None, None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            cap.release()
            return frame, largest_face

    print(f'No face in all frames in {video_path}')
    cap.release()
    return None, None


def get_largest_face_from_image(image_path, face_detector):
    """
    Get the largest face from image.
    Args:
        image_path: (str) image path
        face_detector: dlib face detector
    Returns:
        image: (np.ndarray) frame ndarray in RGB order, if detected, otherwise, return None
        largest_face: (dlib_pybind11.rectangle) detected largest face
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(image, 1)
    if len(faces) == 0:
        print(f'No face in source image: {image_path}')
        return None, None

    largest_face = max(face, key=lambda rect: rect.width() * rect.height())
    return image, largest_face



if __name__ == '__main__':
    face_detector = dlib.get_frontal_face_detector()

    predictor_path = './shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    rgb = cv2.imread('../videos/003/000.png')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb, 1)
    if len(faces):
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        print(face, type(face))
        landmarks = get_key_points(rgb, face, predictor)

        mask = None
        cropped_face, M = image_align_and_crop(rgb, landmarks, (256, 256))
        if mask is not None:
            cropped_mask = mask_align_and_crop(mask, M, (256, 256))

        cropped_face = cropped_face[:, :, ::-1]

        cv2.imwrite('../test_cropped.png', cropped_face)
        print(M, type(M))

        h, w = rgb.shape[:2]
        recover_img = cv2.warpAffine(cropped_face, cv2.invertAffineTransform(M), (w, h))

        cv2.imwrite('../test_recover.png', recover_img)

        gray = cv2.cvtColor(recover_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        result = rgb.copy()
        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(mask))
        result += recover_img
        cv2.imwrite('../test_result.png', result)



