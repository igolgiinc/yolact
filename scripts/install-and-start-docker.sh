#!/bin/bash
#docker load < ig-p1classifier-1.2-10.0.tar.gz 
#docker images

docker network create mcmt_p1classifier_net

# 0
docker stop mcmt_p1classifier_0 mcmt_p1classifier_web0; docker rm mcmt_p1classifier_0 mcmt_p1classifier_web0; 
docker run --network mcmt_p1classifier_net --detach --runtime=nvidia --name mcmt_p1classifier_0 --restart=unless-stopped ig-p1classifier:1.2-10.0-p1class /bin/bash -c "export CUDA_VISIBLE_DEVICES=0 ; cd /root/p1classifier ; python3.6 eval.pyc --trained_model=/root/p1classifier/weights/yolact_base_54_800000.pth --score_threshold=0.5 --top_k=100 --run_with_flask true --contours_json true --flask_max_parallel_frames 20 --use_flask_devserver false"
mcmt_p1classifier_0_ip=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mcmt_p1classifier_0`
echo $mcmt_p1classifier_0_ip
docker run --network mcmt_p1classifier_net --publish 10000:10000 --detach --runtime=nvidia --name mcmt_p1classifier_web0 --restart=unless-stopped ig-p1classifier:1.2-10.0-p1class /bin/bash -c "cd /root/p1classifier ; gunicorn -w 4 --bind 0.0.0.0:10000 'wsgi:run_webapp(zmqserver_ip=\"${mcmt_p1classifier_0_ip}\")' --access-logfile - --error-logfile - --log-level=info;"

# 1
docker stop mcmt_p1classifier_1 mcmt_p1classifier_web1; docker rm mcmt_p1classifier_1 mcmt_p1classifier_web1; 
docker run --network mcmt_p1classifier_net --detach --runtime=nvidia --name mcmt_p1classifier_1 --restart=unless-stopped ig-p1classifier:1.2-10.0-p1class /bin/bash -c "export CUDA_VISIBLE_DEVICES=1 ; cd /root/p1classifier ; python3.6 eval.pyc --trained_model=/root/p1classifier/weights/yolact_base_54_800000.pth --score_threshold=0.5 --top_k=100 --run_with_flask true --contours_json true --flask_max_parallel_frames 20 --use_flask_devserver false"
mcmt_p1classifier_1_ip=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mcmt_p1classifier_1`
echo $mcmt_p1classifier_1_ip
docker run --network mcmt_p1classifier_net --publish 10001:10000 --detach --runtime=nvidia --name mcmt_p1classifier_web1 --restart=unless-stopped ig-p1classifier:1.2-10.0-p1class /bin/bash -c "cd /root/p1classifier ; gunicorn -w 4 --bind 0.0.0.0:10000 'wsgi:run_webapp(zmqserver_ip=\"${mcmt_p1classifier_1_ip}\")' --access-logfile - --error-logfile - --log-level=info;"

