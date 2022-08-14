FROM gitlab/gitlab-runner:alpine as build1


FROM build1 as build2
FROM python:3.7
RUN pip install --upgrade pip
WORKDIR /home/pramod/
COPY requirements.txt /home/pramod/
COPY train_test.py /home/pramod/
COPY configuration /home/pramod/
COPY wrappers /home/pramod/
RUN pip install -r requirements.txt	
RUN git clone https://github.com/cycraig/gym-platform.git
COPY PPO_gym_platform.py /home/pramod/
CMD [ "python3", "./PPO_gym_platform.py"]


