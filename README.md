## Download Dataset

`VoxCeleb1`| https://pan.quark.cn/s/e22a74e8dc8d?pwd=txw7 | 提取码：txw7

`VoxCeleb2`| https://pan.quark.cn/s/38821c5dc835?pwd=4pjq | 提取码：4pjq

`musan.tar.gz`| https://pan.quark.cn/s/d804108f53dc?pwd=zFUm | 提取码：zFUm

`rirs_noises.zip`| https://pan.quark.cn/s/5ae224886d83?pwd=aLyf | 提取码：aLyf

## Download Dataset
All these datasets are backed up on Quark (Philip). If you want to download them, please follow the links and passcodes.
| Dataset                 | Link                                                | 提取码 |
|-------------------------|-----------------------------------------------------|--------|
| VoxCeleb1               | https://pan.quark.cn/s/e22a74e8dc8d?pwd=txw7        | txw7   |
| VoxCeleb2               | https://pan.quark.cn/s/38821c5dc835?pwd=4pjq        | 4pjq   |
| musan.tar.gz            | https://pan.quark.cn/s/632e1933d640?pwd=EFez        | EFez   |
| rirs_noises.zip         | https://pan.quark.cn/s/632e1933d640?pwd=EFez        | EFez   |
| Voxceleb2_proc_segments |                                                     |        |
| merged_labels.txt       |                                                     |        |
| Voxceleb2_proc_segments |                                                     |        |


## Preparation
- Download all the datasets:
    - Download `VoxCeleb1` and `VoxCeleb2`. Extract the `wav` folder to `./dbsource/VoxCeleb1` and `./dbsource/VoxCeleb2` respectively.
    - Extract `musan.tar.gz` to `./dbsource/Others/musan_split`
    - Extract `rirs_noises.zip` to `./dbsource/Others/RIRS_NOISES`
- Preprocess `VoxCeleb2` into segments:
    - Run `1_prep_voxceleb2_proc.py` to preprocess VoxCeleb2. The processed segments will be saved to `./voxceleb2_proc_segments/VoxCeleb2/train/wav`. It will also generate `label_part1.txt`, `label_part2.txt`, and `label_part3.txt` in `./voxceleb2_proc_segments/`. Note the reason to preprocess VoxCeleb2 is to speed up training for ECAPA-TDNN and other audio models.
    - Run `2_merge_voxceleb2_proc_label.py` to merge `label_part1.txt`, `label_part2.txt`, and `label_part3.txt` into `merged_labels.txt`.
    - Done. If we want to use the processed segments for training, we can use `./voxceleb2_proc_segments/VoxCeleb2/train/wav` as the `train_path`. 
    - Note that `3_zip_voxceleb2_proc.py` is not used in this project. It's main purpose is to split the processed segments into multiple zip files and backup the processed segments on Quark.

## Dataset Tree

```tree
dbsource/
...VoxCeleb1/
......train/
.........wav/
............idxxxxxx/
......test/
.........wav/
............idxxxxxx/
......list_test_all.2txt
......list_test_hard2.txt
......veri_test2.txt

...VoxCeleb2/
......train/
.........wav/
............idxxxxxx/
......train_list.txt

...Others/
......musan_split/
.........musan/
............music/
............noise/
............speech/
......RIRI_NOISES/
.........pointsource_noises/
.........real_rirs_isotropic_noises/
.........simulated_rirs/

voxceleb2_proc_segments/
...VoxCeleb2/
......train/
.........wav/
...label_part1.txt
...label_part2.txt
...label_part3.txt
...merged_labels.txt

voxceleb2_proc_zips/ # zip the processed voxceleb2_procsegments into multiple zip files
...voxceleb2_proc_segments_1.zip
...voxceleb2_proc_segments_2.zip
...voxceleb2_proc_segments_3.zip
```