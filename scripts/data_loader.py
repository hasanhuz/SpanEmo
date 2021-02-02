from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class DataClass(Dataset):
    def __init__(self, args, filename):
        self.args = args
        self.filename = filename
        self.max_length = int(args['--max-length'])
        self.data, self.labels = self.load_dataset()

        if args['--lang'] == 'English':
            self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif args['--lang'] == 'Arabic':
            self.bert_tokeniser = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        elif args['--lang'] == 'Spanish':
            self.bert_tokeniser = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

        self.inputs, self.lengths, self.label_indices = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(self.filename, sep='\t')
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        return x_train, y_train

    def process_data(self):
        desc = "PreProcessing dataset {}...".format('')
        preprocessor = twitter_preprocessor()

        if self.args['--lang'] == 'English':
            segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
            label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                           "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
        elif self.args['--lang'] == 'Arabic':
            segment_a = "غضب توقع قرف خوف سعادة حب تفأول اليأس حزن اندهاش أو ثقة؟"
            label_names = ['غضب', 'توقع', 'قر', 'خوف', 'سعادة', 'حب', 'تف', 'الياس', 'حزن', 'اند', 'ثقة']

        elif self.args['--lang'] == 'Spanish':
            segment_a = "ira anticipaciÃ³n asco miedo alegrÃ­a amor optimismo pesimismo tristeza sorpresa or confianza?"
            label_names = ['ira', 'anticip', 'asco', 'miedo', 'alegr', 'amor', 'optimismo',
                           'pesim', 'tristeza', 'sorpresa', 'confianza']

        inputs, lengths, label_indices = [], [], []
        for x in tqdm(self.data, desc=desc):
            x = ' '.join(preprocessor(x))
            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)

            #label indices
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return inputs, labels, length, label_idxs

    def __len__(self):
        return len(self.inputs)
