# EasyInv
This is the official implementation of our paper ["EasyInv: Toward Fast and Better DDIM Inversion"](https://arxiv.org/html/2408.05159v1)
If you are running our codes based on SDV1-4, then
1. Before running the codes, copy three files in folder "replace_diffusers" to ".../site-packages/diffusers/models/" and replace the original files. Rember to make a backup of these replaced files incase you want to run other project in this enviroment.

2. Run "by_referenceSDV1-4.ipynb" for trying our method.

Otherwise, for SDXL version, run "EasyInv_SDXL/by_reference.ipynb".

For whatever version you expect to try, please make sure your own paths in .ipynb files and download dataset and model as following instructions.

1. During runing the code, if you got problem for download the SD pre-train model by "StableDiffusionPipeline.from_pretrained", download it by yourself and place it under "CompVis" folder.

2. Images in "special_cases" folder in the original results of some pictures we presented in our paper

3. Since COCO 2017 is a open source dataset, we submit our image chosen script "choose_img.py" together with codes instead. Anyone who have download the COCO 2017 dataset can extract images we used by this script from the testing and validation set.

4. For testing on your own images, make sure they are name in continues integers starting from 0 and in the type of .jpeg, for example '0.jpeg', '1.jpeg', ......, '999.jpeg'. Besides, all these image shounld be square.

# Implement our method on your own inversion framework
If you have already implement DDIM inversion, please check `example.py` for applying our methods to improve it. As we mentioned, this is really easy.

# Reference
Part of our codes are based on following project:
1. [prompt-to-prompt](https://github.com/google/prompt-to-prompt)
```bibtex
@article{hertz2022prompt,
  title = {Prompt-to-Prompt Image Editing with Cross Attention Control},
  author = {Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2208.01626},
  year = {2022},
}
```

If you feel this project to be useful, please cite and star us! The bibtex citation of our paper us as following.

1. ["EasyInv: Toward Fast and Better DDIM Inversion"](https://arxiv.org/html/2408.05159v1)
```bibtex
@article{zhang2024easyinv,
  title={EasyInv: Toward Fast and Better DDIM Inversion},
  author={Zhang, Ziyue and Lin, Mingbao and Yan, Shuicheng and Ji, Rongrong},
  journal={arXiv preprint arXiv:2408.05159},
  year={2024}
}
```
