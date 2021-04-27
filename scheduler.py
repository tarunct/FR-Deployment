import os
import pickle
import time
import warnings
from rq.decorators import job
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import config as cfg
import dbutils
from logger import get_logger

warnings.filterwarnings('ignore')


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10, 20]
    gammas = [0.001, 0.01, 0.1, 1, 10]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, verbose=5, n_jobs=1)
    grid_search.fit(X, y)
    return grid_search.best_params_


@job('retrain-model')
def retrain_model(json_):
    """

    :param json_:
    :return:
    """
    if json_ is None:
        return 0

    log_file = os.path.join(cfg.dirc['LOGS'], 'model-retrain', 'Scheduler.log')
    logger = get_logger(log_file)

    start_time = time.time()

    application = json_['application']
    groupName = json_['groupId']
    frModelId = json_['frModelId']

    logger.info('New retrain request received, application: {} group: {} modelId: {}'.format(application, groupName, frModelId))

    # function to return userlist of model
    frUserIds = dbutils.get_userlist(frModelId=frModelId)
    logger.info('Users: {}'.format(frUserIds))

    userList, embeddings = dbutils.get_user_embeddings(tuple(frUserIds))

    if application != cfg.DUMMY_APPLICATION_NAME and len(frUserIds) < 3:
        logger.info('Not enough users in the group, current users: {} Adding dummy users!'.format(len(frUserIds)))
        du_userlist, du_embeddings = dbutils.get_dummy_data()
        userList = userList + du_userlist
        embeddings = embeddings + du_embeddings

    le = LabelEncoder()
    labels = le.fit_transform(userList)

    best_params = svc_param_selection(embeddings, labels, 10)
    recognizer = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', probability=True)
    # logger.info("Best C : %d , Best Gamma : %d  " % (best_params['C'], best_params['gamma']))

    recognizer.fit(embeddings, labels)

    model_path = os.path.join(cfg.dirc['RECOGNITION_MODELS'], application, groupName, str(frModelId))

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    recognizer_path = os.path.join(model_path, 'rec.pickle')
    le_path = os.path.join(model_path, 'le.pickle')

    f = open(recognizer_path, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open(le_path, "wb")
    f.write(pickle.dumps(le))
    f.close()

    # Update the UserStatus of users included in the model
    dbutils.set_users_active(tuple(frUserIds))
    dbutils.reset_model_retrain_flag(frModelId)

    logger.info('Total time taken to train models: {}'.format(time.time() - start_time))

    return 1
