# depth data process
## NDS data for depth estimation pipeline

``` 
    feature 1: data process by line
    feature 2: one dateset for one NDS file
    feature 3: using multi process
```


## Dependencies
```
    git clone git@github.com:nreal-ai/bfuncs.git
    cd bfuncs
    python3 setup.py develop
```

## Supported dataset 
```
    NYUv2
    posetrack
    inria
    RedWeb
```

## RUN command
```
    python create_mix_dataset_for_midas.py -o PATH

```


  