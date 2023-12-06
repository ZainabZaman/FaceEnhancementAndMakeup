import os
import cv2
import argparse
import glob
import torch
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.archs.codeformer_arch import CodeFormer

from basicsr.utils.registry import ARCH_REGISTRY

from PIL import Image, ImageDraw
import face_recognition

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

img_name = '/content/drive/MyDrive/test images/image_2023_09_04T05_48_18_589Z.png'
image = face_recognition.load_image_file(img_name)
option_makeup = 1
# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    if option_makeup == 1:
      #DEEP GRAY EB
      d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
      d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
      d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
      d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)

      #RED LP
      d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
      d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
      d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=2)
      d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=2)

      #GRAY ES
      d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
      d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

      #BLACK EL
      d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 80), width=4)
      d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 80), width=4)

    elif option_makeup == 2:
      #BROWN EB
      d.polygon(face_landmarks['left_eyebrow'], fill=(110, 38, 14, 70))
      d.polygon(face_landmarks['right_eyebrow'], fill=(110, 38, 14, 70))
      d.line(face_landmarks['left_eyebrow'], fill=(110, 38, 14, 70), width=1)
      d.line(face_landmarks['right_eyebrow'], fill=(110, 38, 14, 70), width=1)

      #HOT PINK LP
      d.polygon(face_landmarks['top_lip'], fill=(199, 21, 133, 128))
      d.polygon(face_landmarks['bottom_lip'], fill=(199, 21, 133, 128))
      d.line(face_landmarks['top_lip'], fill=(199, 21, 133, 128), width=2)
      d.line(face_landmarks['bottom_lip'], fill=(199, 21, 133, 128), width=2)

      #GRAY ES
      d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
      d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

      #BROWN EL
      d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(139, 69, 19, 100), width=4)
      d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(139, 69, 19, 100), width=4)

    elif option_makeup == 3:
      #DEEP GRAY EB
      d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
      d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
      d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
      d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)

      #DARK ORANGE BROWN LP
      d.polygon(face_landmarks['top_lip'], fill=(210, 105, 30, 128))
      d.polygon(face_landmarks['bottom_lip'], fill=(210, 105, 30, 128))
      d.line(face_landmarks['top_lip'], fill=(210, 105, 30, 128), width=2)
      d.line(face_landmarks['bottom_lip'], fill=(210, 105, 30, 128), width=2)

      #GRAY ES
      d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
      d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

      #BLACK EL
      d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 80), width=4)
      d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 80), width=4)

    elif option_makeup == 4:
      #DEEP GRAY EB
      d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
      d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
      d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
      d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)

      #LIGHT PINK LP
      d.polygon(face_landmarks['top_lip'], fill=(255, 105, 180, 60))
      d.polygon(face_landmarks['bottom_lip'], fill=(255, 105, 180, 60))
      d.line(face_landmarks['top_lip'], fill=(255, 105, 180, 60), width=2)
      d.line(face_landmarks['bottom_lip'], fill=(255, 105, 180, 60), width=2)

      #GRAY ES
      d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
      d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

      #BROWN EL
      d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(139, 69, 19, 100), width=4)
      d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(139, 69, 19, 100), width=4)

    elif option_makeup == 5:
      #DEEP GRAY EB
      d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90))
      d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90))
      d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 90), width=1)
      d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 90), width=1)

      #CRIMSON LP
      d.polygon(face_landmarks['top_lip'], fill=(220, 20, 1, 60))
      d.polygon(face_landmarks['bottom_lip'], fill=(220, 20, 1, 60))
      d.line(face_landmarks['top_lip'], fill=(220, 20, 1, 60), width=2)
      d.line(face_landmarks['bottom_lip'], fill=(220, 20, 1, 60), width=2)

      #GRAY ES
      d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
      d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

      #BLACK EL
      d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 80), width=4)
      d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 80), width=4)

    # pil_image.save("/content/result.jpg")
    # pil_image.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # ------------------------ input & output ------------------------
    detection_model = 'retinaface_resnet50'
    w = 0.7
    upscale = 2
    option_upscale = 2
    result_root = f'results/test_img_{w}'

    # if not args.output_path is None: # set output path
    result_root = './'

    # ------------------ set up CodeFormer restorer -------------------
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)

    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    print(f'Face detection model: {detection_model}')

    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start processing ---------------------
    face_helper.read_image(opencvImage)
    # Get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
    print(f'\tDetect {num_det_faces} faces')
    # Align and warp each face
    face_helper.align_warp_face()

    # Face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # Prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face, cropped_face)

    if option_upscale == 1: #low enhancement
        # Paste back the restored faces
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()
    elif option_upscale == 2: #high enhancement
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()
        brightness = 4
        contrast = 1.2
        restored_img = cv2.addWeighted(restored_img, contrast, np.zeros(restored_img.shape, restored_img.dtype), 0, brightness)

    # Save the restored image
    name = os.path.basename(img_name)
    save_restore_path = os.path.join(result_root, f'restored_{name}')
    imwrite(restored_img, save_restore_path)

    print(f'\nAll results are saved in {result_root}')

