# 360 Image Saliency Prediction by Embedding Self-Supervised Proxy Task

The saliency results can been seen in result folder.

You can download the encoder training (proxy_task) dataset, decoder training (VR-EyeTracking) dataset, validation (Salient360) dataset, encoder weight, and EPSNet weight in the following link: https://drive.google.com/file/d/1J_eycgNpGj5M9IvtTvzER6XNcheeiV46/view?usp=sharing

After download the file, unzip it to the path "abc". Then change the path in the second line of the constants.py file to "abc".


Used the following commands to obtain encoder weight, decoder weight, saliency results, and evaluation results.
```
1. python3 cross_attention_train.py 
```
```
2. python3 supervised_saliency_train.py
```
```
3. python3 supervised_output.py
```
```
4. python3 saliencyMeasure_low_resloution.py 
```

Or you can directly use our provided cross_attention_100.pkl and cross_attention_saliency_100e.pkl, use command 2 or 3, directly train the decoder or get saliency results.

You can also download the VR-EyeTracking dataset by yourself, and generate the saliency map by txt2fix.py and fix2saliency.py.
