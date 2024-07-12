# ChordGen
**A tool to convert a sung melody into music played in any instrument with temperature control on the chord progressions created.**

*12/07/2024 - This model was trained with low compute and on a smaller version of the full dataset. 
Model can be further improved with sufficient compute.

---

## ChordGen Flow

![ChordGen Layout Visualization](/Input/Misc/ChordGenFlow.PNG)



## Run on Unix (or Windows Git Bash)
Before running, ensure that you have added your .wav or .mid file to **"ChordGen/Input"** and that it is the only file in this folder.

Navigate to library location: ```cd __libraryPath__```

To change the execution permissons: ```chmod +x main.sh```

To **run** the ChordGen flow:  ```./main.sh```

---
<details><summary>Training & Inference</summary>

### Train
To train the model on your own data, add your data into the folder **"data/TrainingData"**, insure that the chord and melody files are named according to the name given in the example dataset. Then run the following script to train the model:
```
python model.py train
```
### Inference
If you wish to use the pretrained script or after you have trained your own model, run inference directly on __NoteSequenceInputString_:
```
python model.py inference __NoteSequenceInputString__
```

</details>

---




### Dependancies
* argparse
* mido
* librosa
* midiutil
* csv
* torch

Install these using:
```
pip install __library__
```
