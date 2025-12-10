import torch
from options import TrainOptions
from ssp import ssp
import numpy as np
from PIL import Image
from torchvision import transforms 
import torch
from patch import patch_img
import numpy as np
import cv2
import random as rd
from random import random, choice
from scipy.ndimage import gaussian_filter
from io import BytesIO

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def sample_randint(s):
    if len(s) == 1:
        return s[0]
    return rd.randint(s[0], s[1])


def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_randint(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def processing(img, opt):
    if opt.aug:
        aug = transforms.Lambda(
            lambda img: data_augment(img, opt)
        )
    else:
        aug = transforms.Lambda(
            lambda img: img
        )

    if opt.isPatch:
        patch_func = transforms.Lambda(
            lambda img: patch_img(img, opt.patch_size, opt.trainsize))
    else:
        patch_func = transforms.Resize((256, 256))

    trans = transforms.Compose([
        aug,
        patch_func,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return trans(img)

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.isVal = True
    # blur
    val_opt.blur_prob = 0
    val_opt.blur_sig = [1]
    # jpg
    val_opt.jpg_prob = 0
    val_opt.jpg_method = ['pil']
    val_opt.jpg_qual = [90]
    # if len(val_opt.blur_sig) == 2:
    #     b_sig = val_opt.blur_sig
    #     val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    # if len(val_opt.jpg_qual) != 1:
    #     j_qual = val_opt.jpg_qual
    #     val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def judge(image):
    model = ssp()
    model_path = 'path.pth' 
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  
    model.load_state_dict(state_dict) 
    model.eval()
    with torch.no_grad():
        #image_path = 'images/xxx.jpg' 
        #image = Image.open(image_path)  
        image = image.convert('RGB')
        val_opt = get_val_opt()
        image = processing(image, val_opt)
        image = image.unsqueeze(0) 
        res = model(image)
        res = torch.sigmoid(res).ravel()
        return res < 0.5
        #if res > 0.5:
            #print("Real")
            #return "Real"
        #else:
            #print("Fake")
            #return "Fake"