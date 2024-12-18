export pretrained_path="your_model_path_of_SD1.5"

accelerate launch train.py --opt ./options/cidm/task_1.yml --task_id 1 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_2.yml --task_id 2 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_3.yml --task_id 3 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_4.yml --task_id 4 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_5.yml --task_id 5 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_6.yml --task_id 6 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_7.yml --task_id 7 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_8.yml --task_id 8 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_9.yml --task_id 9 --pretrained_path $pretrained_path
accelerate launch train.py --opt ./options/cidm/task_10.yml --task_id 10 --pretrained_path $pretrained_path