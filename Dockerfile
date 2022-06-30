FROM continuumio/miniconda3

ENV venvName=koamex
WORKDIR /home/workspace
COPY requirements.txt .

RUN conda create --name $venvName python=3 -y
RUN echo "conda activate $venvName" >> ~/.bashrc

RUN /bin/bash -c "source activate koamex && \
conda install albumentations pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y && \
conda install --file requirements.txt -y"

RUN rm requirements.txt

CMD ["/bin/bash"]
