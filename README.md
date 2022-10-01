Using this repo is simple:
1. Clone repo: `git clone https://github.com/AnkushMalaker/easy-stt.git`
2. Create conda environment:
```
conda create -n easy_stt python=3.8
cd easy-stt
pip install -r requirements.txt
```
3. Run inference script and select the model
`python3 src/scripts/infer_live.py -c` to run live inference through a connected microphone
or
`python3 src/scripts/infer.py ./input_file.wav ./output.csv -c` to run inference on `input_file.wav` and save result in `output.csv`.
You can even provide a directory instead of a file in the above command and it'll run inference on all files and save results in the csv.

Note: For `infer_live.py`, the user needs to find which audio device is suitable. Maybe we could build an interface to prompt the user to select the audio device. For now it's manual.


## Results
Audio 1:
https://user-images.githubusercontent.com/43288948/193379872-3d51a2ef-a729-46df-be53-bb0fc9b9cb37.mp4
Transcription: 
WEREBASICALLY TRYING TO RETAIN THE FINAL LAYER OF THE MODEL SO THAT IT CAN RECOGNIZE MY VOICE AND ACCENT AND ME BETTER

Audio 2:
https://user-images.githubusercontent.com/43288948/193379897-2a5d848d-feab-4bda-ad97-60305d43d9c5.mp4
THESE MODELS ARE TRAINED ON LARGE CORPORA THAT DOES N'T ALWAYS TRANSLATE TO GREAT PERFORMANCE IN SPECIFIC CASES
