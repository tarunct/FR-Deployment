#!/bin/bash
path=$(pwd)
echo "$path"
start() {
   echo ".....Face Recognition server is going to start....."
   # shellcheck disable=SC2164
   cd "$path"

   echo "Activating Facerecognition virtual environment"
   source activate Facerecognition

   echo "Starting redis worker for onboard-request queue"
   nohup python3 FR_OnboardingWorker.py &

   echo "Starting redis worker for retrain-model queue"
   nohup python3 FR_RetrainingWorker.py &

   echo "Starting Face recognition api"
   python3 FR_Onboard.py &
   sleep 5
}

stop() {
   echo ".....Face Recognition server is going to stop....."

   echo "Stopping Face recognition api"
   pkill -f -9 FR_Onboard.py

   echo "Shutting onboard-request worker"
   pkill -f -15 'FR_OnboardingWorker.py'

   echo "Shutting retrain-model worker"
   pkill -f -15 'FR_RetrainingWorker.py'

   echo "Killing remaining Facerecognition processes"
   pkill -f -15 Facerecognition

}

case "$1" in
    start)
       start
       ;;
    stop)
       stop
       ;;
    restart)
       stop
       start
       ;;
    *)
       echo "Usage: $0 {start|stop|restart}"
esac