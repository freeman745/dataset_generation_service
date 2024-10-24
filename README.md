# Dataset Generation Service
- image comparison (to find the difference before/after pick)
- image augmentations
- tote image simulation

## Image Comparison

### dataset versions
|       |    train      |    val | 
|-------|---------------|----- |
| v1  | N/A | 1526 | 
| v2  |  |  |

### image comparison versions
v1: base on depth map pixel comparation


### image comparison performance

train/val


|    |    IOU (depth mask)    |  IOU (depth contour) | 
|-------|---------------|----|
| v1  | N/A | /84% | 
| v2  |  |  |


### Build and run

docker build -t dataset_generation_service .

docker run -p 4001:4001 --name dataset_generation_service -v {path you want to save the dataset}:/app/generated_dataset -v /data/db_data/kojiya_221229/Item:/source -d dataset_generation_service


### Interface document

#### Input

| name       | type       | comment       |
| --------- | --------- | --------- |
| date   | string   | i.e: 20240529   |
| order_id   | string   | Order ID   |
| batch_size   | int   | Number of images enough<br>for 1 training session   |
| padding   | float   | Padding for cropping image<br>Default 0.05  |
| template_match   | int   | Use template match or not. 1 for use, 0 for not. Default: 0   |

#### Output

| name       | type       | comment       |
| --------- | --------- | --------- |
| coco_path   | string   | Path to COCO file<br>Empty string for not been created   |
| code   | int   | Error code   |
| msg   | string   | Error message   |
