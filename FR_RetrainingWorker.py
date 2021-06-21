from redis import Redis, StrictRedis
from rq import Queue, Worker
import config as cfg

redis_conn = StrictRedis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, password=cfg.REDIS_PWD)
q = Queue(cfg.REDIS_RETRAIN_QUEUE, connection=redis_conn)


if __name__ == '__main__':
    # Start a worker with a custom name
    worker = Worker([q], connection=redis_conn, name='FR_RetrainingWorker')
    worker.work()
