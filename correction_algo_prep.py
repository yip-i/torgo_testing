
"""Data prep"""
import re
from datasets import load_dataset, load_metric, DatasetDict, Dataset, Audio
from huggingsound import SpeechRecognitionModel
import pandas as pd
import librosa
from tqdm import tqdm

from asr_testing_jonatas import remove_special_characters, prep_csv



def prep_training_data(model, torgo_dataset):
    """
    """
    references = []
    for i in tqdm(range(torgo_dataset.num_rows)):

        prediction = model.transcribe([torgo_dataset[i]["audio"]["path"]])
        prediction = prediction[0]["transcription"]

        row = {"path": torgo_dataset[i]["audio"]["path"],
        "actual": torgo_dataset[i]["text"].lower() ,
        "prediction": prediction}
        references.append(row)
    return references



def main():
    speaker = 'F01'

    model = SpeechRecognitionModel("yip-i/torgo_xlsr_finetune-" + speaker + "-2")
    data = load_dataset('csv', data_files='output.csv')
    data = data.cast_column("audio", Audio(sampling_rate=16_000))   

    timit = data['train'].filter(lambda x: x == speaker, input_columns=['speaker_id'])
    train_data_transcribed = data['train'].filter(lambda x: x != speaker, input_columns=['speaker_id'])

    timit = timit.map(remove_special_characters)
    train_data_transcribed.map(remove_special_characters)


    # test_data = prep_training_data(model, timit)
    train_data = prep_training_data(model, train_data_transcribed)

    print(train_data)

if __name__=="__main__":
    prep_csv("output_og.csv", min_length=1)

    main()