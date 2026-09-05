"""
Byte-Pair Encoding (BPE) Tokenizer
Stage 01 - Challenge 01
"""

from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size: int = 50):
        """
        Initializes the BPE Tokenizer.
        
        Args:
            vocab_size: Target size of the vocabulary (base characters + merged tokens).
        """
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.unk_token = "<unk>"

    def get_stats(self, vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        """
        Counts frequency of adjacent symbol pairs across all words in vocab.
        
        Args:
            vocab: Mapping from word tuple (e.g. ('l', 'o', 'w', '</w>')) to its frequency count.
            
        Returns:
            Dictionary mapping symbol pairs (tuple[str, str]) to total frequency across the corpus.
        """
        # TODO: Implement this function
        # 1. Iterate over (word_tuple, freq) in vocab.items()
        # 2. For each adjacent pair (word_tuple[i], word_tuple[i+1]), aggregate count * freq
        # 3. Return the pairs frequency dictionary
        ans_dict = defaultdict(int)
        for keys in vocab.keys():
            for i in range(len(keys)-1):
                v1=keys[i]
                v2=keys[i+1]
                ans_dict[(v1,v2)]+=vocab[keys];
                # print(vocab[keys])
        # print(ans_dict)
        return ans_dict

        raise NotImplementedError("Implement get_stats")

    def merge_vocab(self, pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
        """
        Merges all occurrences of the given pair in the vocabulary word tuples.
        
        Args:
            pair: Tuple of two consecutive symbols to merge, e.g. ('e', 's').
            vocab: Current vocabulary dictionary mapping word tuples to counts.
            
        Returns:
            New vocabulary dictionary with the specified pair merged into a single symbol string.
        """
        # TODO: Implement this function
        # 1. Create a new vocabulary dict: new_vocab = {}
        # 2. For each word tuple in vocab:
        #    Iterate through the symbols and replace consecutive occurrences of (pair[0], pair[1])
        #    with pair[0] + pair[1].
        # 3. Preserve the word frequency in new_vocab.
        
        new_vocab_keys=list()
        # print(vocab)
        for keys in vocab.keys():
            new_keys=list()
            skip=False
            for i in range(len(keys)):
                if(skip):
                    skip=False
                    continue
                if i<len(keys) and pair[0]==keys[i] and pair[1]==keys[i+1]:
                    new_keys.append(pair[0]+pair[1])
                    skip=True
                else:
                    new_keys.append(keys[i])
            new_vocab_keys.append(tuple(new_keys))
        x=0
        new_vocab=defaultdict(int)
        for value in vocab.values():
            new_vocab[new_vocab_keys[x]]=value
            x+=1
        # print(new_vocab)
        return new_vocab

        raise NotImplementedError("Implement merge_vocab")

    def train(self, corpus: str):
        """
        Trains the BPE tokenizer on the given corpus until vocab_size is reached.
        
        Args:
            corpus: Plain text string used for training.
        """
        # TODO: Implement this function
        # 1. Extract words from corpus (e.g. corpus.strip().split()).
        # 2. Build initial vocab dict where each word is a tuple of characters + '</w>',
        #    e.g. "low" -> ('l', 'o', 'w', '</w>') with its frequency count.
        # 3. Initialize base vocabulary with all unique individual characters + '</w>' + '<unk>'.
        #    Build self.token_to_id and self.id_to_token mappings.
        # 4. Loop while len(self.token_to_id) < self.vocab_size:
        #    a. pairs = self.get_stats(vocab)
        #    b. If pairs is empty, break.
        #    c. Pick the pair with the maximum frequency (tie-breaker: standard max).
        #    d. vocab = self.merge_vocab(best_pair, vocab)
        #    e. Append best_pair to self.merges.
        #    f. Add merged token (best_pair[0] + best_pair[1]) to token_to_id & id_to_token.

        words = corpus.split(' ')
        dictionary = set()
        vocab=defaultdict(int)
        for word in words:
            l = list(word)
            l.append("</w>")
            vocab[tuple(l)]+=1  
        # print(vocab)
        dictionary.add('</w>')
        dictionary.add(self.unk_token)
        # x=0
        # for index, item in enumerate(dictionary):
        #     self.token_to_id[item]=index
        #     self.id_to_token[index]=item
        x=0
        while len(self.token_to_id)<self.vocab_size:
            pairs = self.get_stats(vocab)
            # print(pairs)
            mx=0
            max_pair=tuple()
            for key in pairs.keys():
                if(pairs.get(key)>=mx):
                    mx=pairs.get(key)
                    max_pair=key
            if(len(pairs.keys())==0):
                break
            # print(max_pair)
            # print(vocab)
            vocab = self.merge_vocab(max_pair, vocab)
            # print(vocab)
            self.merges.append(max_pair)
            # x+=1
            # if(x==5):
            #     break

        for key in vocab.keys():
            for i in key:
                dictionary.add(i)

        for index, item in enumerate(dictionary):
            self.token_to_id[item]=index
            self.id_to_token[index]=item
        
        # print(self.token_to_id)
        # raise NotImplementedError("Implement train")
        

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.
        
        Args:
            text: Input string to tokenize.
            
        Returns:
            List of integer token IDs.
        """
        # TODO: Implement this function
        # 1. Split text into words.
        # 2. Convert each word to initial symbol tuple: tuple(list(word) + ['</w>']).
        # 3. For each pair in self.merges (in the exact order they were learned during training):
        #    Merge the pair in all word representations if present.
        # 4. Flatten the tokens from all words and convert each token to its token ID
        #    using self.token_to_id (use self.token_to_id[self.unk_token] if unknown).
        # 5. Return the list of token IDs.
        words = text.split(' ')
        tokens = list()
        for word in words:
            w=word+'</w>'
            if(self.token_to_id.get(w)==None):
                tokens.append(self.token_to_id[self.unk_token])
            else:
                tokens.append(self.token_to_id[w])
        return tokens
        raise NotImplementedError("Implement encode")

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into the original text string.
        
        Args:
            ids: List of integer token IDs.
            
        Returns:
            Decoded string with '</w>' converted to spaces.
        """
        # TODO: Implement this function
        # 1. Convert each ID to its token string using self.id_to_token (or '<unk>').
        # 2. Concatenate all token strings.
        # 3. Replace '</w>' with ' ' and strip extra whitespace.
        # 4. Return reconstructed text.
        s = str()
        for id in ids:
            s+=self.id_to_token.get(id).removesuffix('</w>');

            s+=" "
        
        return s.removesuffix(' ')
        raise NotImplementedError("Implement decode")

# bpet = BPETokenizer()
# bpet.train("hug hug pug pug pun bun bun bun")

# print(bpet.encode("hug pug bun"))
