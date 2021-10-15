import torch
import torch.utils as utils


def rightpad(array, max_len=128):
    if len(array) > max_len:
        padded_array = array[: max_len]
    else:
        padded_array = array + ([0] * (max_len - len(array)))

    return padded_array


def segment(text, segmenter):
    words = []
    for word in segmenter.tokenize(text):
        words = words + word
    text = ' '.join([word for word in words])

    return text


def encode(text, segmenter, vocab):
    text = segment(text, segmenter)
    text = '<s> ' + text + ' </s>'
    text_ids = vocab.encode_line(text, append_eos=False, add_if_not_exist=False).long().tolist()

    return text_ids


'''
class ReviewDataset(utils.data.Dataset):
    def __init__(self, reviews, ratings,
                        segmenter, vocab,
                        tokenizer_bert,
                        tokenizer_xlm):
        self.segmenter = segmenter
        self.vocab     = vocab
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_xlm = tokenizer_xlm
        self.dataset   = [
            (
                rightpad(encode(reviews[i], self.segmenter, self.vocab)),               #PhoBert
                rightpad(self.tokenizer_bert.encode("[CLS] " + reviews[i] + " [SEP]")), #Bert
                rightpad(self.tokenizer_xlm.encode("<s> " + reviews[i] + " </s>")),     #XLM
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_phobert, review_bert, review_xlm, rating = self.dataset[index]
        review_phobert = torch.tensor(review_phobert)
        review_bert = torch.tensor(review_bert)
        review_xlm = torch.tensor(review_xlm)
        rating = torch.tensor(rating)


        return review_phobert, review_bert, review_xlm, rating
'''


class ReviewDataset(utils.data.Dataset):
    def __init__(self, reviews, ratings,
                        segmenter, vocab,
                        tokenizer_bert,
                        tokenizer_lstmcnn):
        self.segmenter = segmenter
        self.vocab     = vocab
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_lstmcnn = tokenizer_lstmcnn


        '''
        for i in range(len(reviews)):
            print(i, reviews[i])
            print('phobert', rightpad(encode(reviews[i], self.segmenter, self.vocab)))
            print('bert', rightpad(self.tokenizer_bert.encode("[CLS] " + reviews[i] + " [SEP]")))
            print('lstmcnn', rightpad(self.tokenizer_lstmcnn.texts_to_sequences([reviews[i]])[0]))

            break
        '''


        self.dataset   = [
            (
                rightpad(encode(reviews[i], self.segmenter, self.vocab)),               #PhoBert
                rightpad(self.tokenizer_bert.encode("[CLS] " + reviews[i] + " [SEP]")), #Bert
                rightpad(self.tokenizer_lstmcnn.texts_to_sequences([reviews[i]])[0]),   #lstmcnn
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_phobert, review_bert, review_lstmcnn, rating = self.dataset[index]
        review_phobert = torch.tensor(review_phobert)
        review_bert = torch.tensor(review_bert)
        review_lstmcnn = torch.tensor(review_lstmcnn)
        rating = torch.tensor(rating)


        return review_phobert, review_bert, review_lstmcnn, rating


class ReviewDatasetBert(utils.data.Dataset):
    def __init__(self, reviews, ratings, tokenizer):
        self.tokenizer = tokenizer
        self.dataset   = [
            (
                rightpad(self.tokenizer.encode("[CLS] " + reviews[i] + " [SEP]")),
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_bert, rating = self.dataset[index]
        review_bert, rating = torch.tensor(review_bert), torch.tensor(rating)


        return review_bert, rating


class ReviewDatasetPhoBert(utils.data.Dataset):
    def __init__(self, reviews, ratings, segmenter, vocab):
        self.vocab     = vocab
        self.segmenter = segmenter
        self.dataset   = [
            (
                rightpad(encode(reviews[i], self.segmenter, self.vocab)),
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_phobert, rating = self.dataset[index]
        review_phobert, rating = torch.tensor(review_phobert), torch.tensor(rating)


        return review_phobert, rating


class ReviewDatasetXLM(utils.data.Dataset):
    def __init__(self, reviews, ratings, tokenizer):
        self.tokenizer = tokenizer
        self.dataset   = [
            (
                rightpad(self.tokenizer.encode("<s> " + reviews[i] + " </s>")),
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_xlm, rating = self.dataset[index]
        review_xlm, rating = torch.tensor(review_xlm), torch.tensor(rating)


        return review_xlm, rating


class ReviewDatasetLSTMCNN(utils.data.Dataset):
    def __init__(self, reviews, ratings, tokenizer):
        self.tokenizer = tokenizer

        for i in range(len(reviews)):
            print(reviews[i])
            print(self.tokenizer.texts_to_sequences(reviews[i]))
            break

        self.dataset   = [
            (
                rightpad(self.tokenizer.texts_to_sequences(reviews[i])),
                ratings[i],
            )
            for i in range(len(reviews))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review_lstmcnn, rating = self.dataset[index]
        review_lstmcnn, rating = torch.tensor(review_lstmcnn), torch.tensor(rating)


        return review_lstmcnn, rating
