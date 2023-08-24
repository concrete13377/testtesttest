Nvidia-docker is needed

Usage:

1. To run generation   
```python run.py path_to_content_image path_to_style_image path_to_dir_store_result```

1. Build an image  
    ```docker build -t testtesttest .```

2. To run 1 time  
    ```docker run --gpus all --shm-size 8GB -v /pathtocode/testtesttest:/app -v /pathtodata/data:/data testtesttest python /app/run.py /data/1.jpg /data/2.jpg /data/output```

3. Or run container detached and run infer several times without removing container  
    ```docker run -d --name test1 --gpus all --shm-size 8GB -v /pathtocode/testtesttest:/app -v /pathtodata/data:/data testtesttest sleep inf```    
    
    ```docker exec -it test1 python /app/run.py /data/1.jpg /data/2.jpg /data/output```

Example:  
 
<img src="1.jpg" width="300">+
<img src="2.jpg" width="145">=
<img src="test1_result.jpg" width="200">  


<img src="3.jpg" width="300">+
<img src="4.jpg" width="150">=
<img src="test2_result.jpg" width="200">