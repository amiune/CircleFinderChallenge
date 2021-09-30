The docker was tested with the following command lines:

docker build -t <id> .
docker run -v <local_data_path>:/data:ro -v <local_writable_area_path>:/wdata -it <id>

Once inside the docker container:

The working folder will be /work

./test.sh /data/test/ /wdata/
This creates the solution geojson files in ./wdata/solution/ and image masks in ./wdata/tmp_masks/
This process should take less than 3 hours in CPU

./train.sh /data/train/
This removes and creates again the model in ./circledetectionModel.pt
This process should take less than 40 hours in CPU