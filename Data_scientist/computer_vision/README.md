# Object Detection in an Urban Environment

Write-up Document

Project Overview:

This project is aimed at detecting different objects in an image of real world traffic. Such a model can be deployed in self driving cars to understand the environment around the vehicle and information of upcoming traffic/objects. We are using tensorflow object detection API’s pre trained model to build our model which learns to detect objects in urban environment

Set up:

Downloading a pre trained model:

•	cd /home/workspace/experiments/pretrained_model/

•	wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

•	tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

•	rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

PS : No need to create a config, we already have it
Model Training:
cd /home/workspace
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

Tensorboard visualization:
python -m tensorboard.main --logdir experiments/reference/

Evaluation:
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/

Export Trained Model:
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/

Extract animation:
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif


Dataset


•	Exploratory Data Analysis:
o	Displaying 10 images with color coding for each class(more details in  Exploratory Data Analysis.ipynb ) 
o	Training data

![image](https://user-images.githubusercontent.com/64709471/206431076-d0c5f7ae-29dd-4d38-88de-3aefec5c470c.png)
o	Similar images are displayed for validation and test sets too
o	Additional analysis of class distribution:
	Training:
![image](https://user-images.githubusercontent.com/64709471/206431245-dee0b7d9-d9ad-417b-b0d3-12ad409ebc74.png)
	Validation data
![image](https://user-images.githubusercontent.com/64709471/206431296-8b07e4c1-48a7-47d7-a217-6fa083e0ca9c.png)
	Testing data
![image](https://user-images.githubusercontent.com/64709471/206431335-c599224f-94c7-4095-9056-e0ee9354d063.png)
•	Explore Image augmentation
•	First run
o	Training
	Reference experiment
![image](https://user-images.githubusercontent.com/64709471/206431383-44263f5f-5d77-45ab-89e7-e0dc1fb45c0e.png)
o	Evaluation
![image](https://user-images.githubusercontent.com/64709471/206431419-42d06598-9af8-4caf-861d-49b37c5cdbc8.png)

Improving on reference:
•	Augmentation applied
o	random_adjust_hue : This is done so that vehicles are detected not based on their colors anad conversely all cars should be deteced as cars irrespective of its color
![image](https://user-images.githubusercontent.com/64709471/206431489-326515d6-6603-4836-b06a-fce550686189.png)
o	adjust_gamma :  Gamma value changes difference between light and dark areas(icreasing gamma would make darker area darker and light areas lighter)
![image](https://user-images.githubusercontent.com/64709471/206431540-c059346b-3603-45de-9f8e-76bdaf555899.png)
o	random_jitter_boxes : this changes box bounds by a value and would help in generalization
![image](https://user-images.githubusercontent.com/64709471/206431583-39325f58-dec9-468d-9270-e982bffa8897.png)


o	Training
	Loss is smaller with all augmentations mentioned above
	Algorithm has started to pick up objects in image as seen in testing
	Low classification and localization loss
	Slow decreasing of loss towards optimal(not yet there)

![image](https://user-images.githubusercontent.com/64709471/206431640-f9ba8d61-f6e1-433d-9244-13261b9adbb0.png)


![image](https://user-images.githubusercontent.com/64709471/206431658-4ddda2f3-4aae-4b6c-83e8-b51ee41b7ada.png)

Other Modifications of tensorflow obejct detection API explored include:
1.	Changing optimizer from momentum optimizer to RMS optimizer
2.	Changing learning rate schedule from cosine decay to exponential decay
3.	increasing batch size is no longer possible with memory usage overhead
4.	changing achor_generator values
5.	IOU threshold for NMS
6.	using drop out and augmentation to avoid overfitting






