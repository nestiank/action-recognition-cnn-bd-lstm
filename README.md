# Video Action Recognition with Pytorch

![language-python][language-python]
<br>
![participants-solo][participants-solo]
<br>
[![institution-korea-university][korea-university-image]][korea-university-cs-url]
![project-urp][project-urp]

> Rough pytorch implementation of "Action Recognition in Video Sequences using Deep Bi-directional LSTM with CNN Features"

### Disclaimer

This implementation is rough. Although this model should work for video action recognition, here should be many errors and/or differences with the original paper.

### Paper citation

> Amin Ullah, Jamil Ahmad, Khan Muhammad, Muhammad Sajjad, and Sung Wook Baik. "Action recognition in video sequences using deep bi-directional LSTM with CNN features." IEEE Access 6 (2017): 1155-1166. doi: 10.1109/ACCESS.2017.2778011.

### More accurate implementation

You can find tensorflow implementation written by the paper author in [Aminullah6264/BidirectionalLSTM](https://github.com/Aminullah6264/BidirectionalLSTM/).

### How to use

Modify options in main.py first.

If using Google Colab, mount Google Drive before running.

> python main.py

### Requirements

> pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

> pip install numpy requests av

### Working period

20 Nov 2020 - 19 Jan 2021 (62 days)

### License and community improvements

  * MIT License
  * If you are interested in fixing errors and/or differences with the original paper, making a pull request is always sincerely welcome.

<!-- Image definitions -->
[korea-university-image]: https://img.shields.io/badge/Institution-Korea%20University-red
[korea-university-cs-url]: http://cs.korea.ac.kr
[project-urp]: https://img.shields.io/badge/Project-URP-00355f
[language-python]: https://img.shields.io/badge/Language-Python-orange
[participants-solo]: https://img.shields.io/badge/Participants-Solo%20Project-7aa3cc
