import base64
import datetime
import os
import shutil
import traceback
import numpy as np
import config as cfg
import cx_Oracle

TABLE_FRUG = cfg.DB_TABLES['USER_GROUP']
TABLE_FRU = cfg.DB_TABLES['USERS']
TABLE_FRM = cfg.DB_TABLES['MODELS']
TABLE_FRE = cfg.DB_TABLES['EMBEDDINGS']

SEQ_FRUG = cfg.DB_SEQUENCES['USER_GROUP']
SEQ_FRU = cfg.DB_SEQUENCES['USERS']
SEQ_FRM = cfg.DB_SEQUENCES['MODELS']
SEQ_FRE = cfg.DB_SEQUENCES['EMBEDDINGS']

DB_USERTAG = 'ML_FR_API'


def create_connection():
    """ create a database connection to the SQLite database
        specified by the db_file
    :param :
    :return: Connection object or None
    """
    conn = None
    try:
        # dsn_tns = cx_Oracle.makedsn('10.72.12.72', 1521, 'ucor')
        # dsn_tns = cx_Oracle.makedsn('10.50.36.65', 1521, 'ucor')
        # conn = cx_Oracle.connect('UCOR_DEV', 'u7t5b4m2', dsn_tns)
        conn = cx_Oracle.connect(cfg.DB_USER, cfg.DB_PWD, cfg.db_dsn_tns)

        # conn = cfg.db_pool.acquire()
        # print(cfg.db_pool)

    except Exception as e:
        # Report DB access error
        print('Can\'t connect to database: {}'.format(repr(e)))
        print(traceback.print_exc())

    return conn


# *-
def clear_database(conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            cur = conn.cursor()

            query = "DELETE FROM {}".format(TABLE_FRE)
            cur.execute(query)

            query = "DELETE FROM {}".format(TABLE_FRU)
            cur.execute(query)

            query = "DELETE FROM {}".format(TABLE_FRM)
            cur.execute(query)

            query = "DELETE FROM {}".format(TABLE_FRUG)
            cur.execute(query)

            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def check_egroup(application, groupName, conn=None):
    cur = None
    eGroupExists = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT COUNT(*) FROM {} WHERE APPLICATION='{}' AND GROUP_NAME='{}'".format(
                TABLE_FRUG,
                application,
                groupName)
            cur = conn.cursor()

            for row in cur.execute(query):
                eGroupExists = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return eGroupExists


# *-
def check_user(application, groupName, userName, conn=None):
    userExists = None
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT COUNT(*) FROM {} frug, {} frm , {} fru WHERE frug.FR_USERGROUP_ID=frm.FR_USERGROUP_ID AND frm.FR_MODEL_ID=fru.FR_MODEL_ID AND frug.APPLICATION='{}' AND frug.GROUP_NAME='{}' AND fru.USERNAME='{}'".format(
                TABLE_FRUG,
                TABLE_FRM,
                TABLE_FRU,
                application,
                groupName,
                userName)
            cur = conn.cursor()

            for row in cur.execute(query):
                userExists = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return userExists


# *-
def insert_egroup(application, eGroup, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query = "INSERT INTO {} (FR_USERGROUP_ID, GROUP_NAME, APPLICATION, CREATED_BY, CREATED_DT, MODIFIED_BY, MODIFIED_DT) VALUES ({}.nextval, '{}', '{}', '{}', '{}', '{}', '{}')".format(
                TABLE_FRUG,
                SEQ_FRUG,
                eGroup,
                application,
                DB_USERTAG,
                ts,
                DB_USERTAG,
                ts
            )

            # print(query)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def insert_frmodel(frGroupId, frModelNo, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query = "INSERT INTO {} (FR_MODEL_ID, FR_USERGROUP_ID, USER_COUNT, MODEL_NO, RETRAIN_FLAG, CREATED_BY, CREATED_DT, MODIFIED_BY, MODIFIED_DT) VALUES ({}.nextval, {}, {}, {}, {}, '{}', '{}', '{}', '{}')".format(
                TABLE_FRM,
                SEQ_FRM,
                frGroupId,
                0,
                frModelNo,
                0,
                DB_USERTAG,
                ts,
                DB_USERTAG,
                ts
            )

            # print(query)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()
        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def insert_user(frModelId, userName, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query = "INSERT INTO {} (FR_USER_ID, FR_MODEL_ID, USERNAME, USER_STATUS, CREATED_BY, CREATED_DT, MODIFIED_BY, MODIFIED_DT) VALUES ({}.nextval, {}, '{}', '{}', '{}', '{}', '{}', '{}')".format(
                TABLE_FRU,
                SEQ_FRU,
                frModelId,
                userName,
                'Profiles Incomplete',
                DB_USERTAG,
                ts,
                DB_USERTAG,
                ts)
            # print(query)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

            update_modelusercount(frModelId=frModelId)

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def assign_frmodel(frGroupId, conn=None):
    cur = None
    frModelId = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT FR_MODEL_ID, MODEL_NO, USER_COUNT FROM {} WHERE FR_USERGROUP_ID={} ORDER BY MODEL_NO".format(
                TABLE_FRM,
                frGroupId)
            cur = conn.cursor()

            for row in cur.execute(query):
                if row[2] < cfg.MODEL_USERS_THRESHOLD:
                    frModelId = row[0]
                    break

            if frModelId is None:
                query = "SELECT MAX(MODEL_NO) FROM {} WHERE FR_USERGROUP_ID={}".format(
                    TABLE_FRM,
                    frGroupId)

                newModelNo = None
                for row in cur.execute(query):
                    newModelNo = row[0]

                newModelNo = newModelNo + 1
                insert_frmodel(frGroupId, newModelNo)

                frModelId = get_frmodelid(frGroupId, newModelNo)

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return frModelId


# *-
def get_frgroupid(application, groupname, conn=None):
    cur = None
    frGroupId = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT FR_USERGROUP_ID FROM {} WHERE APPLICATION='{}' AND GROUP_NAME='{}'".format(
                TABLE_FRUG,
                application,
                groupname)
            cur = conn.cursor()

            for row in cur.execute(query):
                frGroupId = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return frGroupId


# *-
def get_frmodelid(frGroupId, frModelNo, conn=None):
    cur = None
    frModelId = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT FR_MODEL_ID FROM {} WHERE FR_USERGROUP_ID={} AND MODEL_NO={}".format(
                TABLE_FRM,
                frGroupId,
                frModelNo)
            cur = conn.cursor()

            for row in cur.execute(query):
                frModelId = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return frModelId


# *-
def get_fruserid(frModelId, userName, conn=None):
    cur = None
    frGroupId = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT FR_USER_ID FROM {} WHERE FR_MODEL_ID={} AND USERNAME='{}'".format(
                TABLE_FRU,
                frModelId,
                userName)
            cur = conn.cursor()

            for row in cur.execute(query):
                frGroupId = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return frGroupId


# *-
def get_fruserdetails(application, groupName, userName, conn=None):
    cur = None
    frGroupId = None
    frModelId = None
    frUserId = None
    userStatus = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT frug.FR_USERGROUP_ID, frm.FR_MODEL_ID, fru.FR_USER_ID, fru.USER_STATUS FROM {} frug, {} frm , {} fru WHERE frug.FR_USERGROUP_ID=frm.FR_USERGROUP_ID AND frm.FR_MODEL_ID=fru.FR_MODEL_ID AND frug.APPLICATION='{}' AND frug.GROUP_NAME='{}' AND fru.USERNAME='{}'".format(
                TABLE_FRUG,
                TABLE_FRM,
                TABLE_FRU,
                application,
                groupName,
                userName)
            cur = conn.cursor()

            for row in cur.execute(query):
                frGroupId = row[0]
                frModelId = row[1]
                frUserId = row[2]
                userStatus = row[3]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return frGroupId, frModelId, frUserId, userStatus


# *-
def update_modelusercount(frModelId, conn=None):
    cur = None
    userCount = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT COUNT(FR_USER_ID) FROM {} WHERE FR_MODEL_ID={}".format(
                TABLE_FRU,
                frModelId)
            cur = conn.cursor()

            for row in cur.execute(query):
                userCount = row[0]

            print('Usercount: {}'.format(userCount))

            if userCount == 0:
                cleanup_frmodel(frModelId)

            else:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                query = "UPDATE {} SET USER_COUNT={}, MODIFIED_BY='{}', MODIFIED_DT='{}' WHERE FR_MODEL_ID={}".format(
                    TABLE_FRM,
                    userCount,
                    DB_USERTAG,
                    ts,
                    frModelId)
                cur.execute(query)
                conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def cleanup_frmodel(frModelId, conn=None):
    cur = None
    modelNo = None
    application = None
    groupName = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT frug.APPLICATION, frug.GROUP_NAME, frm.MODEL_NO FROM {} frug, {} frm WHERE frug.FR_USERGROUP_ID=frm.FR_USERGROUP_ID and FR_MODEL_ID={}".format(
                TABLE_FRUG,
                TABLE_FRM,
                frModelId)
            cur = conn.cursor()

            for row in cur.execute(query):
                application = row[0]
                groupName = row[1]
                modelNo = row[2]

            print('modelNo: {}'.format(modelNo))

            if modelNo > 1:
                query = "DELETE FROM {} WHERE FR_MODEL_ID={}".format(
                    TABLE_FRM,
                    frModelId)
                cur.execute(query)
                conn.commit()

                model_path = os.path.join(cfg.dirc['RECOGNITION_MODELS'], application, groupName, str(frModelId))
                shutil.rmtree(model_path)

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def delete_user(frModelId, frUserId, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            cur = conn.cursor()

            query = "DELETE FROM {} WHERE FR_USER_ID={}".format(
                TABLE_FRE,
                frUserId)
            cur.execute(query)
            conn.commit()

            query = "DELETE FROM {} WHERE FR_USER_ID={}".format(
                TABLE_FRU,
                frUserId)
            cur.execute(query)
            conn.commit()

            update_modelusercount(frModelId=frModelId)

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    return 1


# *-
def get_userlist(frModelId, conn=None):
    cur = None
    users = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT FR_USER_ID FROM {} WHERE FR_MODEL_ID={} AND USER_STATUS IN ('Onboarding Complete', 'Profiles Complete')".format(
                TABLE_FRU,
                frModelId)
            cur = conn.cursor()

            cur.execute(query)
            users = [i[0] for i in cur.fetchall()]

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return users


# *-
def get_userstatus(frUserId, conn=None):
    cur = None
    userStatus = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT USER_STATUS FROM {} WHERE FR_USER_ID={}".format(
                TABLE_FRU,
                frUserId)
            cur = conn.cursor()

            for row in cur.execute(query):
                userStatus = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return userStatus


# *-
def get_profiles(frUsedId, conn=None):
    cur = None
    profiles = []
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT DISTINCT Orientation FROM {} WHERE FR_USER_ID={}".format(
                TABLE_FRE,
                frUsedId)
            cur = conn.cursor()
            for row in cur.execute(query):
                profiles.append(row[0])

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return profiles


# *-
def get_dummy_data(conn=None):
    cur = None
    userId = []
    embeddings = []
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT fru.USERNAME, fre.EMBEDDING FROM {} fru, {} fre WHERE fru.FR_USER_ID=fre.FR_USER_ID AND fru.USERNAME IN {} AND fre.ORIENTATION IN {}".format(
                TABLE_FRU,
                TABLE_FRE,
                cfg.DUMMY_USERNAMES,
                cfg.FRONT_FACE_PROFILES)
            cur = conn.cursor()

            for row in cur.execute(query):
                userId.append(row[0])
                embeddings.append(np.frombuffer(base64.b64decode(row[1].read()), np.float32))

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return userId, embeddings


# *-
def get_user_embeddings(userIdList, conn=None):
    cur = None
    usernames = []
    embeddings = []
    try:
        if conn is None:
            conn = create_connection()

        try:
            if len(userIdList) == 1:
                query = "SELECT fru.USERNAME, fre.EMBEDDING FROM {} fru, {} fre WHERE fru.FR_USER_ID=fre.FR_USER_ID AND fru.FR_USER_ID={} AND fre.ORIENTATION IN {}".format(
                    TABLE_FRU,
                    TABLE_FRE,
                    userIdList[0],
                    cfg.FRONT_FACE_PROFILES)
            else:
                query = "SELECT fru.USERNAME, fre.EMBEDDING FROM {} fru, {} fre WHERE fru.FR_USER_ID=fre.FR_USER_ID AND fru.FR_USER_ID IN {} AND fre.ORIENTATION IN {}".format(
                    TABLE_FRU,
                    TABLE_FRE,
                    userIdList,
                    cfg.FRONT_FACE_PROFILES)
            cur = conn.cursor()

            for row in cur.execute(query):
                usernames.append(row[0])
                embeddings.append(np.frombuffer(base64.b64decode(row[1].read()), np.float32))

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return usernames, embeddings


# *-
def get_modelno(frModelId, conn=None):
    cur = None
    modelNo = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT MODEL_NO FROM {} WHERE FR_MODEL_ID={}".format(
                TABLE_FRM,
                frModelId)
            cur = conn.cursor()

            for row in cur.execute(query):
                modelNo = row[0]
                break

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return modelNo


# *-
def set_users_active(userList, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if len(userList) == 1:
                query = "UPDATE {} SET USER_STATUS='Onboarding Complete', MODIFIED_BY='{}', MODIFIED_DT='{}' WHERE FR_USER_ID={}".format(
                    TABLE_FRU,
                    DB_USERTAG,
                    ts,
                    userList[0])

            else:
                query = "UPDATE {} SET USER_STATUS='Onboarding Complete', MODIFIED_BY='{}', MODIFIED_DT='{}' WHERE FR_USER_ID IN {}".format(
                    TABLE_FRU,
                    DB_USERTAG,
                    ts,
                    userList)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return 1


# *-
def delete_profile(frUserId, profile, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "DELETE FROM {} WHERE FR_USER_ID={} AND ORIENTATION='{}'".format(
                TABLE_FRE,
                frUserId,
                profile)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def delete_all_profiles(frUsedId, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "DELETE FROM {} WHERE FR_USER_ID={}".format(
                TABLE_FRE,
                frUsedId)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return 1


# *-
def update_user_status(frUserId, newUserStatus, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            query = "UPDATE {} SET USER_STATUS='{}', MODIFIED_BY='{}', MODIFIED_DT='{}' WHERE FR_USER_ID={}".format(
                TABLE_FRU,
                newUserStatus,
                DB_USERTAG,
                ts,
                frUserId)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return 1


# *-
def update_onboarding_status(frUserId):
    userProfiles = get_profiles(frUserId)
    status = 'Profiles Complete'

    for prof in cfg.MANDOTORY_FACE_PROFILES:
        if prof not in userProfiles:
            status = 'Profiles Incomplete'
            break

    update_user_status(frUserId, status)

    return status


# *-
def set_model_retrain_flag(frModelId, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            query = "UPDATE {} SET RETRAIN_FLAG={}, MODIFIED_BY='{}', MODIFIED_DT='{}' WHERE FR_MODEL_ID={}".format(
                TABLE_FRM,
                1,
                DB_USERTAG,
                ts,
                frModelId)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def reset_model_retrain_flag(frModelId, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            query = "UPDATE {} SET RETRAIN_FLAG={}, MODIFIED_BY='{}', MODIFIED_DT='{}', LAST_RETRAIN_TIME='{}' WHERE FR_MODEL_ID={}".format(
                TABLE_FRM,
                0,
                DB_USERTAG,
                ts,
                ts,
                frModelId)
            cur = conn.cursor()
            cur.execute(query)
            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def insert_embeddings(embedding_rows, conn=None):
    cur = None
    try:
        if conn is None:
            conn = create_connection()

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cur = conn.cursor()
            for _r in embedding_rows:
                query = "INSERT INTO {} (FR_USEREMBEDDING_ID, FR_USER_ID, IMAGE_ID, ORIENTATION, AUGMENTATION, CREATED_BY, CREATED_DT, MODIFIED_BY, MODIFIED_DT, EMBEDDING) VALUES({}.nextval, {}, '{}', '{}', '{}', '{}', '{}', '{}', '{}', :blobdata)".format(
                    TABLE_FRE,
                    SEQ_FRE,
                    _r[0],
                    _r[1],
                    _r[2],
                    _r[3],
                    DB_USERTAG,
                    ts,
                    DB_USERTAG,
                    ts
                )
                cur.execute(query, blobdata=_r[4])

            conn.commit()

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


# *-
def get_front_embeddings(frUserId, conn=None):
    cur = None
    embeddings = []
    try:
        if conn is None:
            conn = create_connection()

        try:
            query = "SELECT EMBEDDING FROM {} WHERE FR_USER_ID={} AND ORIENTATION IN {} AND AUGMENTATION='None'".format(
                TABLE_FRE,
                frUserId,
                cfg.FRONT_FACE_PROFILES)
            cur = conn.cursor()
            # print(query)
            for row in cur.execute(query):
                embeddings.append(np.frombuffer(base64.b64decode(row[0].read()), np.float32))

        except Exception as e:
            print('Exception in dbutils: {}'.format(repr(e)))
            print(traceback.print_exc())

    except Exception as e:
        print('Exception in dbutils: {}'.format(repr(e)))
        print(traceback.print_exc())

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return embeddings
