1. Before running the codes, copy three files in folder "replace_diffusers" to ".../site-packages/diffusers/models/" and replace the original files. Rember to make a backup of these replaced files incase you want to run other project in this enviroment.

2. Run "by_referenceSDV1-4.ipynb" for trying our method.

3. During runing the code, if you got problem for download the SD pre-train model by "StableDiffusionPipeline.from_pretrained", download it by yourself and place it under "CompVis" folder.

4. Images in "special_cases" folder in the original results of some pictures we presented in our paper

5. Since COCO 2017 is a open source dataset, we submit our image chosen script "choose_img.py" together with codes instead. Anyone who have download the COCO 2017 dataset can extract images we used by this script from the testing and validation set.
