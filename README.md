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
