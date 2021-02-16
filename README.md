# EmotionRecognition_2D CNN LSTM networks


## Introduction
  According to the nice paper,[Speech emotion recognition using deep 1D & 2D CNN LSTM networks][paper],the 2D CNN LSTM model was built by tensorflow2-keras modul.With training and testing in [EmoDB][EmoDB], the model we built showed the closest conclusion comparead with the paper.


## Requiremrnts
The code should run the enviroment as follow list:
  name|version
  :---:|:---:
  python|3.8
  numpy|1.19.2
  tensorflow|2.2.0
  librosa|0.8.0
  scikit_learn|0.24.1

Before running the code, you sholud set up the enviroment we needed by entering the following command into the terminal: 
  * `pip install -r requirement.txt`  
  
and then verify the parameter of dataset path in __main.py__  
  * `__EmoDB_file_path__ = 'your_dataset_path'`
  
and finally, running!


## Dataset
  You can dowload [Berlin Database of Emotional Speech][EmoDB].

## Reference
  [JianfengZhao,Xiao Mao,Lijiang Chen, Speech emotion recognition using deep 1D & 2D CNN LSTM networks,Biomedical Signal Processing and Control
Volume 47, January 2019, Pages 312-323][paper]

## Contact
  Any issuse should submit directly or send email msp_baiyuhe@163.com.

[EmoDB]: http://emodb.bilderbar.info/docu/
[paper]: https://www.sciencedirect.com/science/article/abs/pii/S1746809418302337

