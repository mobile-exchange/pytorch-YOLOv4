# Project Structure

```
checkpoints
-- *.pth <-- pytorch weights
-- *.weights <-- darknet weights
demo_darknet2pytorch.py <-- convert darknet to pytorch
```

## Usage

```
python demo_darknet2pytorch.py -cfgfile <model_config> -weightfile <darknet_weights> -output <pytorch_output>
```
