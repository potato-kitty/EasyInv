# EasyInv
This is the official implementation of our paper "EasyInv: Toward Fast and Better DDIM Inversion"

1. Before running the codes, copy three files in folder "replace_diffusers" to ".../site-packages/diffusers/models/" and replace the original files. Rember to make a backup of these replaced files incase you want to run other project in this enviroment.

2. Run "by_referenceSDV1-4.ipynb" for trying our method.

3. During runing the code, if you got problem for download the SD pre-train model by "StableDiffusionPipeline.from_pretrained", download it by yourself and place it under "CompVis" folder.

4. Images in "special_cases" folder in the original results of some pictures we presented in our paper

5. Since COCO 2017 is a open source dataset, we submit our image chosen script "choose_img.py" together with codes instead. Anyone who have download the COCO 2017 dataset can extract images we used by this script from the testing and validation set.

# Reference
Part of our codes are based on following two projects:
1. [prompt-to-prompt](https://github.com/google/prompt-to-prompt)
```bibtex
@article{hertz2022prompt,
  title = {Prompt-to-Prompt Image Editing with Cross Attention Control},
  author = {Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2208.01626},
  year = {2022},
}
```
