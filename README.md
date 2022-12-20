"Sand-dust image enhancement using chromatic variance consistency and gamma correction-based dehazing," Sensors
[Jong Ju Jeon](triplej@pusan.ac.kr), [Tae-Hee Park](thpark77@tu.ac.kr), [Il Kyu Eom*](ikeom@pusan.ac.kr)(https://sites.google.com/view/ispl-pnu)

[Paper](https://doi.org/10.3390/s22239048)


### Requirements ###
1. Linux
2. Python (3.10.8)
3. scikit-image (0.19.3)
4. opencv (4.6.0)


### Usage ###
you can just run through
```shell
python Run_Sand-dust_Enhancement.py 
    --input_dir=/path/to/your/dataset/dir/ \
    --output_dir=/path/to/save/results/ \
    --gamma_max=2.0                               # defaults gamma_max=2.0

#python Run_Sand-dust_Enhancement.py --input_dir=/path/to/your/dataset/dir/ --output_dir=/path/to/save/results/ --gamma_max=2.0

```

### Citation ###
Jong Ju Jeon, Tae-Hee Park, Il Kyu Eom,
Sand-dust image enhancement using chromatic variance consistency and gamma correction-based dehazing,
Sensors,
2022,
https://doi.org/10.3390/s22239048.
(https://www.mdpi.com/1424-8220/22/23/9048)
Keywords: sand-dust image enhancement; chromatic variance consistency; dehazing; gamma correction; cross-correlation; color correction

 