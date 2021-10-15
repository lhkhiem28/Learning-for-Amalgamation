import torch
import torch.utils as utils

def rightpad(array, max_len=128):
    if len(array) > max_len:
        padded_array = array[: max_len]
    else:
        padded_array = array + ([0] * (max_len - len(array)))

    return padded_array

class ReviewDataset(utils.data.Dataset):
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
        review, rating = self.dataset[index]
        review = torch.tensor(review)
        rating = torch.tensor(rating)

        return review, rating