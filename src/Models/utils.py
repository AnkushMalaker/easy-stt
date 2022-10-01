from transformers import Wav2Vec2ForCTC, Wav2Vec2ConformerForCTC


def load_model(model_name):
    if "conformer" in model_name:
        return Wav2Vec2ConformerForCTC.from_pretrained(model_name)
    else:
        return Wav2Vec2ForCTC.from_pretrained(model_name)
