import numpy as np
import torch.utils.data as data


class FluxData(data.Dataset):
    r"""Dataset class loading flux data and mapping it to corresponding token IDs. Supports both decoder and encoder-decoder models.

    Attributes:
        input_ids (np.ndarray([data_size, seq_len_in])): Sequences of input IDs.
        max_len (int): Only for decoder. Maximum length of sequences.
        mask (np.ndarray([data_size, seq_len_in])): Only for decoder. Attention masks.
        output_ids (np.array([data_size, seq_len_out])): Only for encoder-decoder. Sequences of output IDs.
        max_len_in (int): Only for encoder-decoder. Maximum length of input sequences.
        max_len_out (int): Only for encoder-decoder. Maximum length of output sequences.
        mask_enc (np.ndarray([data_size, seq_len_in])): Only for encoder-decoder. Input attention masks.
        mask_dec (np.ndarray([data_size, seq_len_out])): Only for encoder-decoder. Output attention masks.
    """

    def __init__(self, flux_file, tokenizer, model):
        r"""Initializes the dataset.

        Args:
            flux_file (str): Path to file containing flux data.
            tokenizer (IntegerTokenizer): Tokenizer function.
            model (str): Type of model.

        Raises:
            AssertionError: If model is not decoder or encoder-decoder.
        """
        super().__init__()
        self.model = model
        self.path_to_data = flux_file
        self.data = np.loadtxt(flux_file, dtype=str)
        if model == "decoder":
            self.input_ids = tokenizer.data_to_token_ids(self.data)
            self.max_len = self.input_ids.shape[-1]
            self.mask = np.where(self.input_ids == tokenizer.dict["<PAD>"], 0, 1)
        elif model == "encoder-decoder":
            split_idx = np.argwhere(self.data[0] == "<TAB>").item()
            self.input_ids = tokenizer.to_ids(self.data[:, :split_idx])
            self.output_ids = tokenizer.to_ids(self.data[:, split_idx+1:])
            self.max_len_in = self.input_ids.shape[-1]
            self.max_len_out = self.output_ids.shape[-1]
            self.mask_enc = np.where(self.input_ids == tokenizer.dict["<PAD>"], 0, 1)
            self.mask_dec = np.where(self.output_ids == tokenizer.dict["<PAD>"], 0, 1)
        else:
            raise AssertionError(
                f"Only decoder and encoder-decoder model support but got {model}."
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):

        if self.model == "decoder":
            return {
                "input_ids": self.input_ids[idx],
                "masks": self.mask[idx],
            }
        else:
            return {
                "input_ids": self.input_ids[idx],
                "output_ids": self.output_ids[idx],
                "mask_enc": self.mask_enc[idx],
                "mask_dec": self.mask_dec[idx],
            }


class DictTokenizer:
    r"""Tokenizer using given dictionary.

    Attributes:
        base (int): Base for integers.
        dict (dict): Dictionary token to token ID.
        dict_reverse(dict): Dictionary token ID to token.
        vocab_size (int): Size of dictionary, i.e. number of possible tokens.
    """

    def __init__(self, dict):
        r"""Initializes the tokenizer.

        Args:
            dict (dict): Dictionary token to token ID.

        """
        self.dict = dict
        self.base = int(list(dict.keys())[-1]) + 1
        self.dict_reverse = {v: k for k, v in self.dict.items()}
        self.vocab_size = len(self.dict)

    def to_ids(self, tokens):
        r"""Maps sequences of tokens to respective tokens IDs.

        Args:
            tokens (np.ndarray([data_size, seq_len])): Input data consisting of sequences of tokens.

        Returns:
            token_ids (np.ndarray([data_size, max_len])): Tokens IDs.
        """
        token_ids = np.array([[self.dict[token] for token in row] for row in tokens])
        return token_ids

    def to_token(self, tokens):
        r"""Maps sequences of token IDs to respective tokens.

        Args:
            tokens(np.ndarray([data_size, max_len])): Token IDs.

        Returns:
            tokens(np.ndarray([data_size, max_lens)): Tokens.
        """
        tokens = np.array(
            [np.array([self.dict_reverse[id] for id in row]) for row in tokens]
        )
        return tokens