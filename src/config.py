import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16 #32
VALID_BATCH_SIZE = 4 #8
EPOCHS = 1
BASE_MODEL_PATH = "../input/bert_base_uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
