export CUDA_VISIBLE_DEVICES=1
export pretrained_path="your_model_path_of_SD1.5"
text_pet_path=./datasets/evaluation_prompts/test_pet.txt
text_plushy_path=./datasets/evaluation_prompts/test_plushy.txt
text_style_path=./datasets/evaluation_prompts/test_style.txt


python inference.py --text_path $text_pet_path --replace_prompt '<dog1> <dog2>' --pretrained_path $pretrained_path
python inference.py --text_path $text_plushy_path --replace_prompt 'yellow rubber <duck1> <duck2> <duck3>' --pretrained_path $pretrained_path
python inference.py --text_path $text_pet_path --replace_prompt '<dog3> <dog4>' --pretrained_path $pretrained_path
python inference.py --text_path $text_pet_path --replace_prompt '<cat1> <cat2>' --pretrained_path $pretrained_path
python inference.py --text_path $text_pet_path --replace_prompt '<cat3> <cat4>' --pretrained_path $pretrained_path
python inference.py --text_path $text_plushy_path --replace_prompt 'red <backpack1> <backpack2>' --pretrained_path $pretrained_path
python inference.py --text_path $text_plushy_path --replace_prompt '<bear1> <bear2> <bear3>' --pretrained_path $pretrained_path
python inference.py --text_path $text_style_path --replace_prompt '<drawing1> <drawing2>' --pretrained_path $pretrained_path
python inference.py --text_path $text_style_path --replace_prompt '<painting1> <painting2>' --pretrained_path $pretrained_path
python inference.py --text_path $text_style_path --replace_prompt '<ink1> <ink2> <ink3>' --pretrained_path $pretrained_path