import numpy as np
import warnings

def to_base(integer, base):
    r"""Maps integers onto base.

    Args:
        integer (int): Integer to be mapped.

    Returns:
        out (np.ndarray([2+log_base(abs(integer))])): Sequence of tokens representing integer in base.

    """
    if integer == 0:
        return np.array(["+", "0"])
    sign_to_token = {0: "+", 1: "+", -1: "-"}
    sign = sign_to_token[np.sign(integer)]
    abs = np.abs(integer)
    out_len = np.floor(np.emath.logn(base, abs)) + 1
    pows = base ** np.flip(np.arange(out_len))
    out = ((abs // pows) % base).astype(int)
    out = np.concatenate([[sign], out.astype(str)])
    return out


def add_padding(tokens, max_len, add_eos=True):
    r"""Adds padding to token sequence.

    Args:
        tokens (np.ndarray([seq_len])): Sequence of tokens.
        max_len (int): Maximum length of token sequences.
    Returns:
        out (np.ndarray([max_len])): Token sequence with added padding tokens on the right.
    """
    padding = np.full(max_len - len(tokens), "<PAD>")
    if add_eos:
        out = np.concatenate([tokens, np.array(["<EOS>"]), padding])
    else:
        out = np.concatenate([tokens, padding])
    return out


def int_to_token(data, base=1000, do_padding=True, max_len_padding=None, add_eos=True):
    tokens = [
        np.concatenate([to_base(integer, base) for integer in row]) for row in data
    ]
    if do_padding:
        if max_len_padding is None:
            max_len_padding = np.max(np.array([len(seq) for seq in tokens]))
        tokens = np.array([add_padding(seq, max_len_padding, add_eos) for seq in tokens])
    bos = np.repeat("<BOS>", len(tokens))[:, None]
    tokens = np.concatenate([bos, tokens], axis=-1)
    return tokens, max_len_padding


def data_to_token(data, split_idx, max_len_padding=None):
    prompts_tok, _ = int_to_token(data[:, :split_idx], do_padding=False)
    prompts_tok = prompts_tok[:,1:]
    fluxes_tok, max_len_padding = int_to_token(data[:, split_idx:], max_len_padding=max_len_padding)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(data), axis=0)
    data = np.concatenate([prompts_tok, tab_tok, fluxes_tok], axis=-1)
    return data, max_len_padding

def validate_token_seq(tokens):
    r"""Validates single token sequence.

    Args:
        tokens (np.ndarray([data_size, max_len])): Tokens.

    Returns:
        bool: True if valid sequence, False if invalid.
    """
    if tokens[0] not in ["-", "+"]:
        return False
    if not tokens[-1].isdigit():
        return False
    for i in range(1, len(tokens)):
        if tokens[i - 1] not in ["-", "+"] and not tokens[i-1].isdigit():
            return False
        if tokens[i - 1] in ["-", "+"] and not tokens[i].isdigit():
            return False
    return True


def token_seq_to_data(tokens, base):
    r"""Maps single token sequence to respective integer sequence.

    Args:
        tokens (np.ndarray([data_size, max_len])): Valid tokens.

    Returns:
        out (list): Integer sequences.

    """
    int_lst = []
    integer = []
    str_to_sgn = {"+": +1, "-": -1}
    for i in range(len(tokens)):
        if tokens[i] in ["-", "+"]:
            sign = str_to_sgn[tokens[i]]
        else:
            integer.append(int(tokens[i]))
        if i == len(tokens) - 1:
            integer = np.array(integer)
            pow = base ** np.flip(np.arange(len(integer)))
            int_lst.append(np.sum(sign * integer * pow))
            integer = []
        elif not tokens[i + 1].isdigit():
            integer = np.array(integer)
            pow = base ** np.flip(np.arange(len(integer)))
            int_lst.append(np.sum(sign * integer * pow))
            integer = []
    out = np.hstack(int_lst)
    return out


def token_to_data(tokens, tokenizer, return_invalid=False):
    r"""Maps tokens to integer sequences. Invalid sequences of tokens are dropped.

    Args:
        tokens (np.ndarray([data_size, max_len])): Tokens.

    Returns:
        data (list): Integer sequences.
    """
    base = tokenizer.base
    valid_mask = np.array([validate_token_seq(seq) for seq in tokens])
    if np.any(~valid_mask):
        warnings.warn(
            f"Invalid sequence(s) detected in IntegerTokenizer.token_ids_to_data: Dropping {(~valid_mask).sum()} sequence(s)."
        )
    if return_invalid:
        tokens_valid = np.where(valid_mask[:,None], tokens, None)
        data = []

        data = np.vstack([ seq if seq[0] == None else token_seq_to_data(seq, base) for seq in tokens_valid])
        return data, valid_mask
    else:
        tokens_valid = tokens[valid_mask]
        data = np.vstack([token_seq_to_data(seq, base) for seq in tokens_valid])
        return data


def int_nums_to_token(data, base=1000):
    tokens = [
        np.concatenate([to_base(integer, base) for integer in row]) for row in data
    ]
    bos = np.repeat("<INT_START>", len(tokens))[:, None]
    tokens = np.concatenate([bos, tokens], axis=-1)
    return tokens.flatten()

def a_matrix_to_token(data, base=1000):
    tokens =  np.concatenate([to_base(integer, base) for integer in data])
    bos = np.array(["<A_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def b_vector_to_token(data, base=1000):
    tokens =  np.concatenate([to_base(integer, base) for integer in data])
    bos = np.array(["<B_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def hodge_numbers_to_token(data, base=1000):
    tokens =  np.concatenate([to_base(integer, base) for integer in data])
    bos = np.array(["<H_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def chi_to_token(data, base=1000):
    tokens = to_base(data, base)
    bos = np.array(["<C_START>"])
    tokens = np.concatenate([bos, tokens], axis=0)
    return tokens

def top_data_to_token(data, prompt_dic, max_len_padding=None):
    fluxes = data[:,2:]
    prompts = data[:,:2]
    prompts = np.array([prompt_dic[str(pair.astype(int).tolist())] for pair in prompts])
    fluxes_tok, max_len_padding = int_to_token(fluxes, max_len_padding=max_len_padding)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(fluxes_tok), axis=0)
    tok = np.concatenate([prompts, tab_tok, fluxes_tok], axis=-1)
    return tok, max_len_padding

def int_and_mod_to_token(fluxes, moduli, tau, base=1000, max_len_padding=None):
    tokens = []

    for i in range(len(fluxes)):
        flux_tok = np.concatenate([to_base(integer, base) for integer in fluxes[i]])
        mod_tok = np.concatenate([to_base(integer, base) for integer in moduli[i]])
        tau_tok = np.concatenate([to_base(integer, base) for integer in tau[i]])
        tok = np.concatenate([flux_tok, np.array(["<M_START>"]), mod_tok, np.array(["<T_START>"]), tau_tok])
        tokens.append(tok)
    if max_len_padding is None:
        max_len_padding = np.max(np.array([len(seq) for seq in tokens]))
    tokens = np.array([add_padding(seq, max_len_padding) for seq in tokens])
    bos = np.repeat("<BOS>", len(tokens))[:, None]
    tokens = np.concatenate([bos, tokens], axis=-1)
    return tokens, max_len_padding


def mod_data_to_token(prompts, fluxes, moduli, tau, base=1000, max_len_padding_in=None, max_len_padding_out=None):
    prompts_tok, max_len_padding_in = int_to_token(prompts, base=base, max_len_padding=max_len_padding_in, add_eos=False)
    prompts_tok = prompts_tok[:,1:]
    out_tok, max_len_padding_out = int_and_mod_to_token(fluxes, moduli, tau, base, max_len_padding_out)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(prompts), axis=0)
    data = np.concatenate([prompts_tok, tab_tok, out_tok], axis=-1)
    return data, max_len_padding_in, max_len_padding_out


def topo_and_mod_data_to_token(prompt, fluxes, moduli, tau, base=1000, max_len_padding_in=None, max_len_padding_out=None):
    if max_len_padding_in is not None:
        prompt = add_padding(prompt, max_len_padding_in, False)
    prompts_tok = np.repeat(prompt[None,:], len(fluxes), axis=0)
    out_tok, max_len_padding_out = int_and_mod_to_token(fluxes, moduli, tau, base, max_len_padding_out)
    tab_tok = np.repeat(np.array(["<TAB>"])[None, :], len(prompts_tok), axis=0)
    data = np.concatenate([prompts_tok, tab_tok, out_tok], axis=-1)
    return data, max_len_padding_in, max_len_padding_out