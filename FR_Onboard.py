import base64
import io
import logging
import os
import pickle
import statistics
import sys
import time
import cv2
import numpy as np
from PIL import Image
from bob.ip.qualitymeasure import compute_msu_iqa_features
from cerberus import Validator
from flask import Flask, request
from flask import json
from redis import Redis
from rq import Queue
from sklearn.externals import joblib
from DBUpdate import onboard_request
import dbutils
import config as cfg
import warnings
from logger import get_logger

warnings.filterwarnings('ignore')

app = Flask(__name__)


def avg_euc(frModelId, userName, emb):
    frUserId = dbutils.get_fruserid(frModelId=frModelId, userName=userName)
    embs = dbutils.get_front_embeddings(frUserId)
    dist = []

    for _e in embs:
        _d = (np.linalg.norm(emb - _e))
        dist.append(_d)

    app.logger.info(dist)
    euc = statistics.mean(dist)

    return float(euc)


def image_checks(image, profile):
    if np.any(image):
        (h, w) = image.shape[:2]
        image_flip = cv2.flip(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_flip = cv2.cvtColor(image_flip, cv2.COLOR_BGR2GRAY)

        # Check for brightness
        mean_pixel_value = np.mean(gray)

        if mean_pixel_value < 80:
            app.logger.warning('low brightness')
            return -1

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
            app.logger.warning('low face detection confidence')
            return -1

        # Face cropping
        box = face_detections[0, 0, max_confi_ind, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face_cropped = image[startY:endY, startX:endX]
        (fH, fW) = face_cropped.shape[:2]

        # Partial face detected
        if startX < 0 or endX > w or startY < 0 or endY > h:
            app.logger.warning('partial face detected')
            return -1

        # profile detection
        faces_left = profile_detector.detectMultiScale(gray, 1.3, 5)
        faces_right = profile_detector.detectMultiScale(gray_flip, 1.3, 5)

        if profile == 'front' or profile == 'gfront':
            if len(faces_left) > 0 or len(faces_right) > 0:
                app.logger.warning('Front face not detected')
                return -1

        if profile == 'left' and len(faces_left) <= 0:
            app.logger.warning('left face not detected')
            return -1

        if profile == 'right' and len(faces_right) <= 0:
            app.logger.warning('right face not detected')
            return -1

        # Face too far
        if fH < 120 or fW < 120:
            app.logger.warning('face too far')
            return -1

        # Face too close
        if fH > 320 or fW > 320:
            app.logger.warning('face too close')
            return -1


def recognise_response_generator(tokenNo, application, groupId, userId, imageCounter, start_time, start_dtime,
                                 responseCode, responseMessage, spoofStatus='NA', spoofConfidence=0, spoofType='NA',
                                 recognitionStatus='NA', recognitionConfidence=0.0, eucd=0.0):
    response = {'tokenNo': tokenNo, 'application': application, 'groupId': groupId, 'userId': userId,
                'imageCounter': imageCounter, 'recognitionStatus': recognitionStatus,
                'recogniseConfidence': recognitionConfidence, 'euclideanDistance': eucd, 'spoofStatus': spoofStatus,
                'spoofConfidence': spoofConfidence, 'spoofType': spoofType, 'responseCode': responseCode,
                'responseMessage': responseMessage}

    end_time = time.time()
    end_dtime = time.strftime('%m-%d-%y %H:%M:%S', time.localtime(end_time))
    time_diff = end_time - start_time

    response['requestReceiveTs'] = start_dtime
    response['responseSentTs'] = end_dtime
    response['totalExecutionTime'] = time_diff

    app.logger.info(response)
    return response


def onboard_response_generator(responseId, responseCode, responseMessage, imgCount=None, images=None):
    response = {'responseId': responseId, 'responseCode': responseCode, 'responseMessage': responseMessage,
                'acceptedImageCount': imgCount, 'imageResponse': images}

    return response


# Function to extract spoof detection features from the Image
def get_spoof_features(img):
    """
        Generate spoof detection features
        :param img: single image file in OpenCv default format ( BGR, m*n*3)
        :return: image tranformed to RGB channels with dimensions 3*m*n
        """
    # Converting default BGR of OpenCV to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Changing dimension from 3*m*n to m*n*3
    h_im = img[:, :, 0]
    s_im = img[:, :, 1]
    v_im = img[:, :, 2]

    img = np.array([h_im, s_im, v_im])

    return compute_msu_iqa_features(img)


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


# Spoof feature for new model
def get_spoof_features2(img):
    """
        Generate spoof detection features
        :param img: single image file in OpenCv default format ( BGR, m*n*3)
        :return:
        """
    roi = img  # img_bgr[y:y+h, x:x+w]

    img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
    img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

    ycrcb_hist = calc_hist(img_ycrcb)
    luv_hist = calc_hist(img_luv)

    feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
    feature_vector = feature_vector.reshape(1, len(feature_vector))

    return feature_vector


# Store login request image for logs
def store_recognition_image(img, token, application, groupid, userid, imgCounter):
    app.logger.info('image shape: {}'.format(img.shape))
    ts = time.time()
    file_name = '{}_{}_{}_{}_{}_{}.png'.format(ts, token, application, groupid, userid, imgCounter)
    file_path = os.path.join(cfg.dirc['LOGS'], 'login-images', file_name)
    cv2.imwrite(file_path, img)


@app.route('/statuschange', methods=['POST'])
def statuschange():
    json_ = request.get_json(force=True)

    schema = {
        'requestId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'applicationName': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'groupId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'userId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'changeUserStatus': {
            'type': 'string',
            'empty': False,
            'nullable': False,
            'allowed': ['DELETE', 'RESET']
        },
    }

    validator = Validator(schema)
    validator.allow_unknown = True

    Response_json = {}
    ResponseCode = ''
    ResponseMessage = ''

    # Request validation
    if not (validator.validate(json_)):
        errors_list = validator.errors

        for key in errors_list.keys():
            if 'empty values' in errors_list[key][0]:
                ResponseCode = '5001'
                ResponseMessage = 'Missing mandatory field'
                app.logger.error('Missing mandatory field')

            if 'type' in errors_list[key][0]:
                ResponseCode = '5002'
                ResponseMessage = 'Invalid input field type'
                app.logger.error('Invalid input field type')

            if 'unallowed' in errors_list[key][0]:
                ResponseCode = '5003'
                ResponseMessage = 'Invalid input field value'
                app.logger.error('Invalid input field value')

            Response_json['responseId'] = 'R_{}'.format(json_['requestId'])
            Response_json['responseCode'] = ResponseCode
            Response_json['responseMessage'] = ResponseMessage

            return json.dumps(Response_json), {'Content-Type': 'application/json'}

    Response_json['responseId'] = 'R_{}'.format(json_['requestId'])

    application = json_['applicationName']
    eGroup = json_['groupId']
    user = json_['userId']
    newUserStatus = json_['changeUserStatus']

    app.logger.info(
        'New StatusChange({}) request, requestId: {}, application: {}'.format(newUserStatus, json_['requestId'],
                                                                              application))

    if dbutils.check_egroup(application, eGroup) == 0:
        ResponseCode = '2001'
        ResponseMessage = 'User Group not found'
        app.logger.error('User Group not found')

        Response_json['responseCode'] = ResponseCode
        Response_json['responseMessage'] = ResponseMessage
        return json.dumps(Response_json), {'Content-Type': 'application/json'}

    if dbutils.check_user(application, eGroup, user) == 0:
        ResponseCode = '2002'
        ResponseMessage = 'User not found'
        app.logger.error('User not found')

        Response_json['responseCode'] = ResponseCode
        Response_json['responseMessage'] = ResponseMessage
        return json.dumps(Response_json), {'Content-Type': 'application/json'}

    updateResult = 0
    frGroupId, frModelId, frUserId, userStatus = dbutils.get_fruserdetails(application=application, groupName=eGroup,
                                                                           userName=user)

    if newUserStatus == 'DELETE':
        updateResult = dbutils.delete_user(frModelId=frModelId, frUserId=frUserId)

    elif newUserStatus == 'RESET':
        updateResult = dbutils.delete_all_profiles(frUsedId=frUserId)

    if updateResult == 1:
        ResponseCode = '0000'
        ResponseMessage = 'Success'
        app.logger.error('Success')

    elif updateResult == 0:
        ResponseCode = '1000'
        ResponseMessage = 'DB Error'
        app.logger.error('DB Error')

    Response_json['responseCode'] = ResponseCode
    Response_json['responseMessage'] = ResponseMessage

    return json.dumps(Response_json), {'Content-Type': 'application/json'}


@app.route('/statusquery', methods=['POST'])
def statusquery():
    json_ = request.get_json(force=True)

    schema = {
        'requestId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'applicationName': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'users': {
            'type': 'list',
            'empty': False,
            'nullable': False,
            'schema': {
                'type': 'dict',
                'schema': {
                    'groupId': {
                        'type': 'string',
                        'empty': False,
                        'nullable': False
                    },
                    'userId': {
                        'type': 'string',
                        'empty': False,
                        'nullable': False
                    },
                }
            }
        }
    }

    validator = Validator(schema)
    validator.allow_unknown = True

    Response_json = {}
    ResponseCode = ''
    ResponseMessage = ''

    # Request validation
    if not (validator.validate(json_)):
        errors_list = validator.errors

        for key in errors_list.keys():
            if 'empty values' in errors_list[key][0]:
                ResponseCode = '5001'
                ResponseMessage = 'Missing mandatory field'
                app.logger.error('Missing mandatory field')

            if 'type' in errors_list[key][0]:
                ResponseCode = '5002'
                ResponseMessage = 'Invalid input field'
                app.logger.error('Invalid input field')

            Response_json['responseId'] = 'R_{}'.format(json_['requestId'])
            Response_json['responseCode'] = ResponseCode
            Response_json['responseMessage'] = ResponseMessage
            Response_json['users'] = None

            return json.dumps(Response_json), {'Content-Type': 'application/json'}

    Response_json['responseId'] = 'R_{}'.format(json_['requestId'])

    application = json_['applicationName']
    usersResponse = []

    app.logger.info('New Status Query request, requestId: {}, application: {}'.format(json_['requestId'], application))

    for user_dict in json_['users']:
        groupId = user_dict['groupId']
        userId = user_dict['userId']

        if dbutils.check_user(application, groupId, userId) == 1:
            # Generate response for user
            frGroupId, frModelId, frUserId, userStatus = dbutils.get_fruserdetails(application=application,
                                                                                   groupName=groupId, userName=userId)

            userProfiles = dbutils.get_profiles(frUsedId=frUserId)

            app.logger.info(userStatus)
            app.logger.info(userProfiles)

            user_dict = {
                "groupId": groupId,
                "userId": userId,
                "userFound": "True",
                "userStatus": userStatus,
            }

            for prof in cfg.ALL_FACE_PROFILES:
                if prof in userProfiles:
                    user_dict["{}Profile".format(prof)] = 'Complete'
                else:
                    user_dict["{}Profile".format(prof)] = 'InComplete'

            usersResponse.append(user_dict)
        else:
            # Generate response when user doesn't exist
            user_dict = {
                "groupId": groupId,
                "userId": userId,
                "userFound": "False",
                "userStatus": "NA",
                "frontProfile": "NA",
                "leftProfile": "NA",
                "rightProfile": "NA",
                "topProfile": "NA",
                "bottomProfile": "NA",
                "gfrontProfile": "NA"
            }

            usersResponse.append(user_dict)

    ResponseCode = '0000'
    ResponseMessage = 'Completed'

    Response_json['responseId'] = 'R_{}'.format(json_['requestId'])
    Response_json['responseCode'] = ResponseCode
    Response_json['responseMessage'] = ResponseMessage
    Response_json['users'] = usersResponse

    return json.dumps(Response_json), {'Content-Type': 'application/json'}


@app.route('/onboard', methods=['POST'])
def onboard():
    json_ = request.get_json(force=True)

    schema = {
        'requestId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'requestType': {
            'type': 'string',
            'empty': False,
            'nullable': False,
            'allowed': ['New', 'Update']
        },
        'applicationName': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'groupId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'userId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'profile': {
            'type': 'string',
            'empty': False,
            'nullable': False,
            'allowed': cfg.ALL_FACE_PROFILES
        },
        'imageCount': {
            'type': 'integer',
            'empty': False,
            'nullable': False
        },
        'images': {
            'type': 'list',
            'empty': False,
            'nullable': False,
            'schema': {
                'type': 'dict',
                'schema': {
                    'imageId': {
                        'type': 'string',
                        'empty': False,
                        'nullable': False
                    },
                    'imageData': {
                        'type': 'string',
                        'empty': False,
                        'nullable': False
                    },
                }
            }
        }
    }

    validator = Validator(schema)
    validator.allow_unknown = True

    ResponseCode = ''
    ResponseMessage = ''
    ResponseId = 'R_{}'.format(json_['requestId'])

    # Request validation
    if not (validator.validate(json_)):
        errors_list = validator.errors

        for key in errors_list.keys():
            if 'empty values' in errors_list[key][0]:
                ResponseCode = '5001'
                ResponseMessage = 'Missing mandatory field'
                app.logger.error('Missing mandatory field')

            if 'type' in errors_list[key][0]:
                ResponseCode = '5002'
                ResponseMessage = 'Invalid input field type'
                app.logger.error('Invalid input field type')

            if 'unallowed' in errors_list[key][0]:
                ResponseCode = '5003'
                ResponseMessage = 'Invalid input field value'
                app.logger.error('Invalid input field value')

            Response_json = onboard_response_generator(responseId=ResponseId, responseCode=ResponseCode,
                                                       responseMessage=ResponseMessage)
            return json.dumps(Response_json), {'Content-Type': 'application/json'}

    application = json_['applicationName']
    eGroup = json_['groupId']
    user = json_['userId']
    requestType = json_['requestType']
    profile = json_['profile']
    images = json_['images']

    app.logger.info(
        '{} Onboard request, application: {} GroupId: {} UserId: {} Profile: {}'.format(requestType, application,
                                                                                        eGroup, user, profile))

    if requestType == "Update":
        if dbutils.check_egroup(application, eGroup) == 0:
            ResponseCode = '2001'
            ResponseMessage = 'User Group not found'
            app.logger.error('User Group not found')

            Response_json = onboard_response_generator(responseId=ResponseId, responseCode=ResponseCode,
                                                       responseMessage=ResponseMessage)
            return json.dumps(Response_json), {'Content-Type': 'application/json'}

        if dbutils.check_user(application, eGroup, user) == 0:
            ResponseCode = '2002'
            ResponseMessage = 'User not found'
            app.logger.error('User not found')

            Response_json = onboard_response_generator(responseId=ResponseId, responseCode=ResponseCode,
                                                       responseMessage=ResponseMessage)
            return json.dumps(Response_json), {'Content-Type': 'application/json'}

    if requestType == "New":

        if dbutils.check_user(application, eGroup, user) != 0:
            frGroupId, frModelId, frUserId, userStatus = dbutils.get_fruserdetails(application=application,
                                                                                   groupName=eGroup, userName=user)

            if (
                    userStatus == 'Onboarding Complete' or userStatus == 'Profiles Complete') and profile in cfg.MANDOTORY_FACE_PROFILES:
                ResponseCode = '2003'
                ResponseMessage = 'Invalid request'
                app.logger.error('Invalid request')

                Response_json = onboard_response_generator(responseId=ResponseId, responseCode=ResponseCode,
                                                           responseMessage=ResponseMessage)
                return json.dumps(Response_json), {'Content-Type': 'application/json'}

    images_accepted = []
    accepted_image_count = 0
    images_response = []

    ind = 0

    while ind < len(images) and accepted_image_count < cfg.ONBOARDING_IMAGE_THRESHOLD:
        img_dict = images[ind]

        img_response = {'imageId': img_dict['imageId']}

        try:
            image_64 = base64.b64decode(img_dict['imageData'])
            image = np.array(Image.open(io.BytesIO(image_64)))

            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            norm_img = np.zeros((image.shape[0], image.shape[1]))
            norm_img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
            image = norm_img

            img_check = image_checks(image, profile)

            if img_check == -1:
                img_response['imageStatus'] = 'Check Failed'

            else:
                img_response['imageStatus'] = 'Accepted'
                accepted_image_count += 1
                images_accepted.append({'imageId': img_dict['imageId'], 'image': image})

        except Exception as e:
            img_response['imageStatus'] = 'Corrupt'
            app.logger.error('Corrupt Image')

        images_response.append(img_response)
        ind += 1

    app.logger.error('Images accepted: {}'.format(accepted_image_count))

    while ind < len(images):
        img_dict = images[ind]
        img_response = {'imageId': img_dict['imageId'], 'imageResponse': 'NA'}
        images_response.append(img_response)
        ind += 1

    # Set ResponseCode and ResponseMessage
    if accepted_image_count == cfg.ONBOARDING_IMAGE_THRESHOLD:
        ResponseCode = '0000'
        ResponseMessage = 'Success'
        app.logger.info('Success')

        # Send required data to DBUpdate Method
        dbupdate_json = {
            'requestType': requestType,
            'applicationName': application,
            'groupId': eGroup,
            'userId': user,
            'profile': profile,
            'images': images_accepted
        }

        # Send onboarding request data to DBUpdate Redis Queue
        job = q.enqueue(onboard_request, args=(dbupdate_json,))

    else:
        ResponseCode = '1000'
        ResponseMessage = 'Threshold Not Met'
        app.logger.info('Threshold Not Met')

    # Set response JSON
    Response_json = onboard_response_generator(responseId=ResponseId, responseCode=ResponseCode,
                                               responseMessage=ResponseMessage, imgCount=accepted_image_count,
                                               images=images_response)

    return json.dumps(Response_json), {'Content-Type': 'application/json'}


@app.route('/recognise', methods=['POST'])
def recognise():
    json_ = request.get_json(force=True)

    start_time = time.time()
    start_dtime = time.strftime('%m-%d-%y %H:%M:%S', time.localtime(start_time))

    schema = {
        'tokenNo': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'application': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'groupId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'userId': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'base64Image': {
            'type': 'string',
            'empty': False,
            'nullable': False
        },
        'imageCounter': {
            'type': 'integer',
            'empty': False,
            'nullable': False
        },
        'imageShape': {
            'type': 'list',
            'empty': False,
            'nullable': False,
            'schema': {
                'type': 'integer',
            }
        },
        'suspiciousFlag': {
            'type': 'integer',
            'empty': False,
            'nullable': True,
            'required': False
        }
    }

    v = Validator(schema)
    v.allow_unknown = True

    Response_json = {}
    ResponseCode = ''
    ResponseMessage = ''

    if not (v.validate(json_)):
        errors_list = v.errors

        for key in errors_list.keys():

            if 'empty values' in errors_list[key][0]:
                ResponseCode = '5001'
                ResponseMessage = 'Missing mandatory field'

            if 'type' in errors_list[key][0]:
                ResponseCode = '5002'
                ResponseMessage = 'Invalid input field'

    tokenNo = json_['tokenNo']
    application = json_['application']
    eGroup = json_['groupId']
    user = json_['userId']
    imageCounter = json_['imageCounter']

    if 'suspiciousFlag' in json_:
        suspiciousFlag = json_['suspiciousFlag']
    else:
        suspiciousFlag = 1

    app.logger.info(
        'New recognition request, application: {}, user group: {}, user: {}'.format(application, eGroup, user))

    # Invalid Query
    if ResponseCode != "":
        response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup, userId=user,
                                                imageCounter=imageCounter, start_time=start_time,
                                                start_dtime=start_dtime, responseCode=ResponseCode,
                                                responseMessage=ResponseMessage)

        return json.dumps(response), {'Content-Type': 'application/json'}

    # User not found
    if dbutils.check_user(application=application, groupName=eGroup, userName=user) == 0:
        response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup, userId=user,
                                                imageCounter=imageCounter, start_time=start_time,
                                                start_dtime=start_dtime, responseCode='1002',
                                                responseMessage='Person not found')

        return json.dumps(response), {'Content-Type': 'application/json'}

    frGroupId, frModelId, frUserId, userStatus = dbutils.get_fruserdetails(application=application, groupName=eGroup,
                                                                           userName=user)

    rec_model_path = os.path.join(cfg.dirc['RECOGNITION_MODELS'], application, eGroup, str(frModelId))

    # SACode Not Present
    rec_path = os.path.join(rec_model_path, 'rec.pickle')
    le_path = os.path.join(rec_model_path, 'le.pickle')

    if not os.path.exists(rec_path):
        response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup, userId=user,
                                                imageCounter=imageCounter, start_time=start_time,
                                                start_dtime=start_dtime, responseCode='1001',
                                                responseMessage='Model load error')

        return json.dumps(response), {'Content-Type': 'application/json'}

    # Get the model and encoder name
    recogniser = pickle.loads(open(rec_path, "rb").read())
    labelencoder = pickle.loads(open(le_path, "rb").read())

    # Model not trained for user
    if user not in labelencoder.classes_:
        response = recognise_response_generator(tokenNo=tokenNo,
                                                application=application,
                                                groupId=eGroup,
                                                userId=user,
                                                imageCounter=imageCounter,
                                                start_time=start_time,
                                                start_dtime=start_dtime,
                                                responseCode='1003',
                                                responseMessage='Model not trained for user')

        return json.dumps(response), {'Content-Type': 'application/json'}

    # Base64 image decoding error
    try:

        image_64 = base64.b64decode(json_['base64Image'])
        image = np.array(Image.open(io.BytesIO(image_64)))

        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Image normalization
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        norm_img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

        if cfg.FLAG_STORE_RECOGNITION_IMAGES == 1:
            store_recognition_image(image, tokenNo, application, eGroup, user, imageCounter)

    except Exception as e:
        response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup, userId=user,
                                                imageCounter=imageCounter, start_time=start_time,
                                                start_dtime=start_dtime, responseCode='2001',
                                                responseMessage='Image corrupted')

        print(repr(e))
        return json.dumps(response), {'Content-Type': 'application/json'}

    if np.any(norm_img):

        # --------  Spoof Detection Here  --------#
        spoof_feat = [get_spoof_features(image)]
        spoof_feat2 = get_spoof_features2(image)

        photoSpoofProb = photoSpoofClf.predict_proba(spoof_feat2)[0][1]
        videoSpoofProb = videoSpoofClf.predict_proba(spoof_feat)[:, 1][0]

        print('Spoof probabilities; photo: {}, video: {}'.format(photoSpoofProb, videoSpoofProb))

        if suspiciousFlag == 0:
            photoSpoofCutoff = cfg.PHOTO_SPOOF_CUTOFF_NONSUS
            videoSpoofCutoff = cfg.VIDEO_SPOOF_CUTOFF_NONSUS

        else:
            photoSpoofCutoff = cfg.PHOTO_SPOOF_CUTOFF_SUS
            videoSpoofCutoff = cfg.VIDEO_SPOOF_CUTOFF_SUS

        print('Suspicious Flag={}; Spoof cut-offs; photo: {}, video: {}'.format(suspiciousFlag, photoSpoofCutoff, videoSpoofCutoff))

        photoSpoofPred = photoSpoofProb >= photoSpoofCutoff
        videoSpoofPred = videoSpoofProb >= videoSpoofCutoff

        if photoSpoofPred:
            response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup,
                                                    userId=user, imageCounter=imageCounter, start_time=start_time,
                                                    start_dtime=start_dtime, responseCode='4001',
                                                    responseMessage='Spoof detected', spoofStatus='Spoof',
                                                    spoofConfidence=photoSpoofProb, spoofType='Photo')

            return json.dumps(response), {'Content-Type': 'application/json'}

        if videoSpoofPred:
            response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup,
                                                    userId=user, imageCounter=imageCounter, start_time=start_time,
                                                    start_dtime=start_dtime, responseCode='4001',
                                                    responseMessage='Spoof detected', spoofStatus='Spoof',
                                                    spoofConfidence=videoSpoofProb, spoofType='Video')

            return json.dumps(response), {'Content-Type': 'application/json'}

        # --------    Face Recognition Here   --------#
        faceBlob = cv2.dnn.blobFromImage(norm_img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        face_embedder.setInput(faceBlob)
        vec = face_embedder.forward()

        # perform classification to recognize the face
        predictions = recogniser.predict_proba(vec)[0]
        j = np.argmax(predictions)
        recognition_prob = predictions[j]
        recognised_name = labelencoder.classes_[j]

        app.logger.info('Request for: {}, Recognised with: {}'.format(user, recognised_name))

        spoof_prob = max(photoSpoofProb, videoSpoofProb)

        if recognised_name in cfg.DUMMY_USERNAMES:
            response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup,
                                                    userId=user, imageCounter=imageCounter, start_time=start_time,
                                                    start_dtime=start_dtime, responseCode='0000',
                                                    responseMessage='Success', spoofStatus='Non Spoof',
                                                    spoofConfidence=spoof_prob, spoofType='NA',
                                                    recognitionStatus='Not Recognised')

            return json.dumps(response), {'Content-Type': 'application/json'}

        # get euclidean distance
        eucd = avg_euc(frModelId, recognised_name, vec.flatten())

        app.logger.info('EUCD with user {}: {}'.format(recognised_name, eucd))

        if recognised_name == user:
            response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup,
                                                    userId=user, imageCounter=imageCounter, start_time=start_time,
                                                    start_dtime=start_dtime, responseCode='0000',
                                                    responseMessage='Success', spoofStatus='Non Spoof',
                                                    spoofConfidence=spoof_prob, spoofType='NA',
                                                    recognitionStatus='Recognised',
                                                    recognitionConfidence=round(recognition_prob, 2), eucd=eucd)

            return json.dumps(response), {'Content-Type': 'application/json'}

        elif recognised_name != user:
            response = recognise_response_generator(tokenNo=tokenNo, application=application, groupId=eGroup,
                                                    userId=user, imageCounter=imageCounter, start_time=start_time,
                                                    start_dtime=start_dtime, responseCode='0000',
                                                    responseMessage='Success', spoofStatus='Non Spoof',
                                                    spoofConfidence=spoof_prob, spoofType='NA',
                                                    recognitionStatus='Not Recognised',
                                                    recognitionConfidence=round(recognition_prob, 2), eucd=eucd)

            return json.dumps(response), {'Content-Type': 'application/json'}


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 8081

    # Model Paths
    face_detector_path = cfg.dirc['FACE_DETECTER']
    face_embedder_path = cfg.dirc['FACE_EMBEDDER']
    spoof_classifier_path = cfg.dirc['SPOOF_MODEL']

    # Load Face Detectors
    print("[INFO] loading face detector...")
    protoPath = os.path.join(face_detector_path, "deploy.prototxt")
    modelPath = os.path.join(face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel")
    face_detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load Face Profile Detector
    print("[INFO] loading face profile detector...")
    profile_detector = cv2.CascadeClassifier(os.path.join(face_detector_path, 'haarcascade_profileface.xml'))

    # Load Face Embedder
    print("[INFO] loading face embedder...")
    face_embedder = cv2.dnn.readNetFromTorch(os.path.join(face_embedder_path, "openface_nn4.small2.v1.t7"))

    # Load Spoof Detection models
    print("[INFO] loading spoof models...")
    videoSpoofClf = joblib.load(os.path.join(spoof_classifier_path, "video_spoof_clf"))
    photoSpoofClf = joblib.load(os.path.join(spoof_classifier_path, "print-attack_ycrcb_luv_extraTreesClassifier.pkl"))

    # creating log dirs
    os.makedirs(os.path.join(cfg.dirc['LOGS'], 'api'), exist_ok=True)
    os.makedirs(os.path.join(cfg.dirc['LOGS'], 'login-images'), exist_ok=True)
    os.makedirs(os.path.join(cfg.dirc['LOGS'], 'model-retrain'), exist_ok=True)
    os.makedirs(os.path.join(cfg.dirc['LOGS'], 'onboard-request'), exist_ok=True)

    log_file = os.path.join(cfg.dirc['LOGS'], 'api', 'frapi.log')
    app.logger = get_logger(log_file)
    app.logger.setLevel(logging.DEBUG)

    redis_conn = Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
    q = Queue(cfg.REDIS_ONBOARD_QUEUE, connection=redis_conn)

    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
