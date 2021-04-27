from redis import Redis
from rq import Queue, Worker
import config as cfg

redis_conn = Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
q = Queue(cfg.REDIS_ONBOARD_QUEUE, connection=redis_conn)

# Start a worker with a custom name
worker = Worker([q], connection=redis_conn, name='FR_OnboardingWorker')
worker.work()
