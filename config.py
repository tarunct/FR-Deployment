from cryptography.fernet import Fernet
import yaml
import cx_Oracle

dirc = {
    "FACE_DETECTER": "data/dir-ro/Detector",
    "FACE_EMBEDDER": "data/dir-ro/Embedder",
    "SPOOF_MODEL": "data/dir-ro/Spoof",
    "RECOGNITION_MODELS": "data/dir-rw/trained-models",
    "LOGS": "data/dir-rw/logs"
}

ALL_FACE_PROFILES = ('front', 'left', 'right', 'top', 'bottom', 'gfront')
MANDOTORY_FACE_PROFILES = ('front', 'left', 'right')
FRONT_FACE_PROFILES = ('front', 'gfront')
ONBOARDING_IMAGE_THRESHOLD = 8
MODEL_USERS_THRESHOLD = 10

DUMMY_USERNAMES = ('DummyUser-183', 'DummyUser-189')
DUMMY_APPLICATION_NAME = 'DUMMYAPP-01'
DUMMY_GROUP_NAME = 'DUMMYGROUP-01'

DB_TABLES = {
    'USER_GROUP': 'FR_USERGROUPS',
    'USERS': 'FR_USERS',
    'MODELS': 'FR_MODELS',
    'EMBEDDINGS': 'FR_USEREMBEDDINGS'
}

DB_SEQUENCES = {
    'USER_GROUP': 'FR_USERGROUPS_SEQ',
    'USERS': 'FR_USERS_SEQ',
    'MODELS': 'FR_MODELS_SEQ',
    'EMBEDDINGS': 'FR_USEREMBEDDINGS_SEQ'
}

FLAG_STORE_ONBOARDING_IMAGES = 0
FLAG_STORE_RECOGNITION_IMAGES = 1

PHOTO_SPOOF_CUTOFF = 0.4
VIDEO_SPOOF_CUTOFF = 0.9

ENCRYPTION_KEY = b'j6pxRDXGFYKdiO1QcusknYAxtx51bpKkW78k-5ONM_Y='
cipher_suite = Fernet(ENCRYPTION_KEY)


def encrypt(str):
    ciphered_text = cipher_suite.encrypt(str)
    return ciphered_text


def decrypt(ciphered_text):
    unciphered_text = (cipher_suite.decrypt(bytes(ciphered_text, 'utf-8')))
    return bytes(unciphered_text).decode("utf-8")


with open(r'dbconfig.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    params = yaml.safe_load(file)


DB_IP = decrypt(params.get('db_host_ip'))
DB_PORT = decrypt(params.get('db_port'))
DB_TAG = decrypt(params.get('db_tag'))
DB_USER = decrypt(params.get('db_user'))
DB_PWD = decrypt(params.get('db_pwd'))

REDIS_HOST = params.get('redis_host')
REDIS_PORT = params.get('redis_port')
REDIS_ONBOARD_QUEUE = params.get('onboarding_queue')
REDIS_RETRAIN_QUEUE = params.get('retraining_queue')

dsn_tns = cx_Oracle.makedsn(DB_IP, int(DB_PORT), DB_TAG)
db_pool = cx_Oracle.SessionPool(DB_USER, DB_PWD, dsn_tns, min=2, max=5, increment=1, threaded=True)
