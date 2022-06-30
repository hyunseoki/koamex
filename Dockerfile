FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV venvName=koamex
WORKDIR /home/workspace
COPY requirements.txt .

RUN apt-get update -y
RUN apt-get install -y git
RUN pip install -r requirements.txt

# RUN /bin/bash -c "source activate koamex && \
# conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c $

# RUN /bin/bash -c "source activate koamex && \
# pip3 install -r requirements.txt -y"

#conda install albumentations pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 $
# pip3 install --upgrade pip && \
# pip3 install -r requirements.txt -y"

RUN rm requirements.txt

CMD ["/bin/bash"]