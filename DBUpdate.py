import base64
import os
import time
import warnings
from datetime import datetime
import cv2
import numpy as np
import skimage.transform
import skimage.transform
import tensorflow as tf
from redis import Redis
from rq import Queue
from rq.decorators import job
import config as cfg
import dbutils
from scheduler import retrain_model
from logger import get_logger

warnings.filterwarnings('ignore')

# Model Paths
face_detector_path = cfg.dirc['FACE_DETECTER']
face_embedder_path = cfg.dirc['FACE_EMBEDDER']

# Load Face Detectors
protoPath = os.path.join(face_detector_path, "deploy.prototxt")
modelPath = os.path.join(face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel")
face_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load Face Profile Detector
profile_detector = cv2.CascadeClassifier(os.path.join(face_detector_path, 'haarcascade_profileface.xml'))

# Load Face Embedder
face_embedder = cv2.dnn.readNetFromTorch(os.path.join(face_embedder_path, "openface_nn4.small2.v1.t7"))


def generate_embedding(image):
    try:
        if np.any(image):
            (h, w) = image.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0),
                                              swapRB=False, crop=False)

            # Face detections
            face_detector.setInput(imageBlob)
            face_detections = face_detector.forward()

            max_confi = 0.0
            max_confi_ind = 0
            for i in range(0, face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence > max_confi:
                    max_confi = confidence
                    max_confi_ind = i

            # Low Face Detection Confidence
            if max_confi < 0.8:
                return -1

            # Face cropping
            box = face_detections[0, 0, max_confi_ind, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_cropped = image[startY:endY, startX:endX]

            faceBlob = cv2.dnn.blobFromImage(face_cropped, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            face_embedder.setInput(faceBlob)
            vec = face_embedder.forward().flatten()

            if type(vec) == int:
                return -1
            return base64.b64encode(vec)

    except:
        return -1


def augment_flip(img):
    """
    Flip the image horizontally
    :param img:
    :return:
    """
    flip = np.fliplr(img)
    return flip


def augment_rotation(img):
    """
    Rotate the image clockwise and anticlockwise by 15 degree
    :param img:
    :return:
    """
    rotate_anti = skimage.transform.rotate(img, angle=15, mode='constant', preserve_range=True).astype(np.uint8)
    rotate_clock = skimage.transform.rotate(img, angle=-15, mode='constant', preserve_range=True).astype(np.uint8)
    return rotate_anti, rotate_clock


def augment_scale(img):
    """
    Scale in and scale out the image
    :param img:
    :return:
    """
    (height, width) = img.shape[:2]
    scalein = skimage.transform.rescale(img, scale=2.0, mode='constant', multichannel=True, preserve_range=True)
    scalein_s = tf.image.central_crop(scalein, 0.5)
    scaleout = skimage.transform.rescale(img, scale=0.5, mode='constant', multichannel=True, preserve_range=True)
    scaleout_s = tf.image.pad_to_bounding_box(scaleout, int(height / 4), int(width / 4), height, width)
    scalein_s = tf.keras.preprocessing.image.img_to_array(scalein_s, data_format=None, dtype=None)
    scaleout_s = tf.keras.preprocessing.image.img_to_array(scaleout_s, data_format=None, dtype=None)
    return scalein_s, scaleout_s


def augment_gaussian_noise(img):
    """
    Add gaussian noise to the image
    :param img:
    :return:
    """
    img = img.astype(np.float32)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=1.0, dtype=tf.float32)
    gaussian_img = tf.add(img, noise)

    gaussian_img = tf.keras.preprocessing.image.img_to_array(gaussian_img, data_format=None, dtype=None)
    return gaussian_img


def augment_translation(img):
    """
    Shift image to left and right side inside frame
    :param img:
    :return:
    """
    pad_left = 100
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    (height, width) = img.shape[:2]

    x = tf.image.pad_to_bounding_box(img, pad_top, pad_left, height + pad_bottom + pad_top,
                                     width + pad_right + pad_left)
    trans_img_left = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)
    x = tf.image.pad_to_bounding_box(img, pad_top, 0, height, width + 100)
    trans_img_right = tf.image.crop_to_bounding_box(x, 0, 100, height, width)

    trans_img_left = tf.keras.preprocessing.image.img_to_array(trans_img_left, data_format=None, dtype=None)
    trans_img_right = tf.keras.preprocessing.image.img_to_array(trans_img_right, data_format=None, dtype=None)
    return trans_img_left, trans_img_right


def augment_brightness(img):
    """
    Generate images with increased and decreased brightness
    :param img:
    :return:
    """
    img_bright = cv2.convertScaleAbs(img, alpha=1.25, beta=0)
    img_dark = cv2.convertScaleAbs(img, alpha=0.75, beta=0)

    return img_bright, img_dark


def store_augmentations(logger, frUserId, profile, img, img_id):
    logger.info('{} - Processing {} profile image for {} image: {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                                                           profile, frUserId, img_id))
    img_flip = augment_flip(img)
    img_scalein, img_scaleout = augment_scale(img)
    img_gaussian = augment_gaussian_noise(img)
    img_trans_left, img_trans_right = augment_translation(img)
    img_bright, img_dark = augment_brightness(img)

    embedding_rows = []
    emb_none = generate_embedding(img)
    emb_flip = generate_embedding(img_flip)
    emb_scaleout = generate_embedding(img_scaleout)
    emb_gaussian = generate_embedding(img_gaussian)
    emb_transleft = generate_embedding(img_trans_left)
    emb_transright = generate_embedding(img_trans_right)
    emb_bright = generate_embedding(img_bright)
    emb_dark = generate_embedding(img_dark)

    if emb_none == -1:
        logger.error(
            '{} - Error in generating embedding for image'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             img_id,
             profile,
             'None',
             emb_none))

    if emb_flip == -1:
        logger.error(
            '{} - Error in generating embedding for flipped image'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_FLIP'.format(img_id),
             profile,
             'FLIP',
             emb_flip))

    if emb_scaleout == -1:
        logger.error('{} - Error in generating embedding for scaleout image'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_SCALEOUT'.format(img_id),
             profile,
             'SCALEOUT',
             emb_scaleout))

    if emb_gaussian == -1:
        logger.error('{} - Error in generating embedding for gaussian image'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_GAUSSIAN'.format(img_id),
             profile,
             'GAUSSIAN',
             emb_gaussian))

    if emb_transleft == -1:
        logger.error('{} - Error in generating embedding for transleft image'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_TRANSLEFT'.format(img_id),
             profile,
             'TRANSLEFT',
             emb_transleft))

    if emb_transright == -1:
        logger.error('{} - Error in generating embedding for transright image'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_TRANSRIGHT'.format(img_id),
             profile,
             'TRANSRIGHT',
             emb_transright))

    if emb_bright == -1:
        logger.error(
            '{} - Error in generating embedding for bright image'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_BRIGHT'.format(img_id),
             profile,
             'BRIGHT',
             emb_bright))

    if emb_dark == -1:
        logger.error(
            '{} - Error in generating embedding for dark image'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    else:
        embedding_rows.append(
            (frUserId,
             '{}_DARK'.format(img_id),
             profile,
             'DARK',
             emb_dark))

    dbutils.insert_embeddings(embedding_rows)


@job('onboard-request')
def onboard_request(json_):
    if json_ is None:
        return 0

    log_file = os.path.join(cfg.dirc['LOGS'], 'onboard-request', 'DBUpdate.log')

    logger = get_logger(log_file)

    start_time = time.time()

    requestType = json_['requestType']
    application = json_['applicationName']
    eGroup = json_['groupId']
    user = json_['userId']
    profile = json_['profile']

    logger.info(
        '{} - {} request for {}-{}-{} profile: {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), requestType,
                                                          application, eGroup, user, profile))

    frGroupId = None
    frModelId = None
    frUserId = None

    if requestType == 'New':
        # dbutils.delete_profile(application, eGroup, user, profile)

        if dbutils.check_egroup(application, eGroup) == 0:
            # create first model for usergroups
            dbutils.insert_egroup(application=application, eGroup=eGroup)
            frGroupId = dbutils.get_frgroupid(application, eGroup)
            dbutils.insert_frmodel(frGroupId=frGroupId, frModelNo=1)

        frGroupId = dbutils.get_frgroupid(application, eGroup)

        if dbutils.check_user(application=application, groupName=eGroup, userName=user) == 0:
            frModelId = dbutils.assign_frmodel(frGroupId)
            dbutils.insert_user(frModelId, user)
            dbutils.update_modelusercount(frModelId)

        frGroupId, frModelId, frUserId, userStatus = dbutils.get_fruserdetails(application, eGroup, user)

    elif requestType == 'Update':
        frGroupId, frModelId, frUserId, userStatus = dbutils.get_fruserdetails(application, eGroup, user)

        if (
                userStatus == 'Onboarding Complete' or userStatus == 'Profiles Complete') and profile in cfg.MANDOTORY_FACE_PROFILES:
            dbutils.delete_all_profiles(frUsedId=frUserId)

        else:
            dbutils.delete_profile(frUserId=frUserId, profile=profile)

    # Generate augmentation of images and insert into table
    for img_dict in json_['images']:
        img = img_dict['image']
        img_id = img_dict['imageId']

        # changes in store_aumentations
        store_augmentations(logger, frUserId, profile, img, img_id)

    userStatus = dbutils.update_onboarding_status(frUserId)

    if userStatus == 'Profiles Complete' and application != cfg.DUMMY_APPLICATION_NAME:
        dbutils.set_model_retrain_flag(frModelId)

        logger.info('{} - Sending model retrain request for {}-{}, modelid: {}'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            application, eGroup, frModelId))

        modelRetrainDict = {'application': application, 'groupId': eGroup, 'frModelId': frModelId}

        redis_conn = Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
        q = Queue(cfg.REDIS_RETRAIN_QUEUE, connection=redis_conn)

        # Send model retrain request data to model-retrain Redis Queue
        job = q.enqueue(retrain_model, args=(modelRetrainDict,))
        redis_conn.connection_pool.disconnect()

    logger.info('Total time taken: {}'.format(time.time() - start_time))
    return 1
