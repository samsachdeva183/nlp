import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = GOOD_SYMBOLS_RE.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS])
    return text.strip()
    
    
def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    # Create a dictionary with default value 0
    # So that all out of vocab words with come to this "unk" tag and idx 0
   
    # Create mappings from tokens (or tags) to indices and vice versa.
    # At first, add special tokens (or tags) to the dictionaries.
    # The first special token must have index 0.
    
    # Mapping tok2idx should contain each token or tag only once. 
    # special_tokens = ['<UNK>', '<PAD>']
    
    i = 0
    vocab = set([t for ts in tokens_or_tags for t in ts])
    vocab_size = len(vocab)+len(special_tokens)
    
    tok2idx = defaultdict(lambda: 0)    
    idx2tok = ['']*vocab_size
    
    for t in special_tokens:
        tok2idx[t] = i
        idx2tok[i] = t
        i +=1
                
    for t_list in tokens_or_tags:
        
        for w in t_list:
            
            if w not in tok2idx:
                tok2idx[w] = i
                idx2tok[i] = w
                i+=1
    
    return tok2idx, idx2tok  
    
    
 def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Output vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
    """
  
    
    if max_len is None:
        columns = max(map(len,batch_captions))
    
    else:
        columns = min(max_len, max(map(len, batch_captions)))
        

    matrix = np.empty([len(batch_captions),columns],'int32')
    
    matrix.fill(pad_idx)

    for i in range(len(batch_captions)):
        line_ix = batch_captions[i][:max_len]
        matrix[i,:len(line_ix)] = line_ix
    
    
    return matrix