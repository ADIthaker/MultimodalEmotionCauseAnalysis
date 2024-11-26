import json
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
#batch
#encode names and emotions, dialutt

def find_sublist_index(main_list, sublist):
    for i in range(len(main_list) - len(sublist) + 1):
        if main_list[i:i + len(sublist)] == sublist:
            return i
    return -1

def join_2d(xs):
    if (len(xs) == 0):
        return []
    elif (len(xs) == 1):
        return xs[0]
    else:
        res = xs[0]
        for i in range(1, len(xs)):
            res += xs[i]
        return res



TagIds = {
    "B" : 1,
    "I" : 2,
    "O" : 0}

#Special
B_TARGET = "[B_TARGET]"
E_TARGET = "[E_TARGET]"
EmotionsTokens = {'anger':'[ANGER]', 
                  'disgust':'[DISGUST]',
                  'fear':'[FEAR]', 
                  'joy':'[JOY]',
                  'sadness':'[SAD]',
                  'surprise':'[SURPISE]',
                  'neutral': '[NEUTRAL]'}
   
EMPTY_LABEL = [TagIds["O"], TagIds["O"], TagIds["O"]]               

class Dataset:
    def __init__(self, path, loadPath = None, train_split = 0.8):
        
        try:
            f = open(loadPath)
            loadedData = json.load(f)
            self.samples = loadedData["samples"]
            self.labels = loadedData["labels"]
            self.attention_masks = loadedData["att_masks"]
        except:
            f = open(path)
            self.data = json.load(f)
            self.splitIdx = int(len(self.data) * train_split)
            self.samples, self.labels,self.attention_masks = self._tokenize(self.data)
        
        
    def qTokenize(self, xs):
        inputs = tokenizer(xs, padding=True, truncation=True,
                           return_tensors="pt", add_special_tokens=False)
        return inputs["input_ids"]
    
    def _tokenize(self, data):
        samples, labels = [], []
        
        excluded = 0
        for scene in self.data:
            this_sample = []
            predUtt = scene['emotion_utterance_ID']
            predUtt = int(predUtt[predUtt.find("utt")+3:]) - 1
            for i,c in enumerate(scene['conversation']):
                res = [EmotionsTokensId[c['emotion']]] + self.qTokenize(c['text'])[0].tolist()
                if i == predUtt:
                    this_sample.append([B_TARGET_ID] + res + [E_TARGET_ID,SEP_ID])
                else:
                    this_sample.append(res + [SEP_ID])
            
            this_label = [[TagIds["O"]] * len(s) for s in this_sample]
            for c in scene['cause_spans']:
                idx = c.find("_")
                uttIdx = int(c[:idx]) -1
                cspan=self.qTokenize(c[idx+1:])[0].tolist()
              
                subuttIdx = find_sublist_index(this_sample[uttIdx], cspan)
           
                assert subuttIdx != -1
                this_label[uttIdx][subuttIdx] = TagIds["B"]
                for i in range(subuttIdx+1, subuttIdx+len(cspan)):
                    this_label[uttIdx][i] = TagIds["I"]
                
            #add cls token
            this_sample[0].insert(0, CLS_ID)
            this_label[0].insert(0, TagIds["O"])
            
            #cut off after the predict
            s_joined = join_2d(this_sample)
            l_joined = join_2d(this_label)
            
            if len(s_joined) > 512:
                excluded += 1
                continue
            
            idxdLabel = []
            for l in l_joined:
                if l == 0:
                    idxdLabel.append([1,0,0])
                elif l == 1:
                    idxdLabel.append([0,1,0])
                else:
                    idxdLabel.append([0,0,1])
                        
            
            assert len(s_joined) == len(l_joined)
            samples.append(s_joined)
            labels.append(idxdLabel)
            assert len(samples) == len(labels)
            
        maxLength = -1
        for s in samples:
            if len(s) > maxLength:
                maxLength = len(s)
        
        attention_masks = []
        #PAD
        for i,s in enumerate(samples):
            pads = maxLength - len(s)
            
            mask = ([False] * len(s)) + ([True] * (pads))
            attention_masks.append(mask)
            
            s += [PAD_ID] * (pads)
            
            
            labels[i] += [[1,0,0]] * (pads)
            
            assert (len(labels[i]) == maxLength)
            assert (len(samples[i]) == maxLength)
            assert (len(attention_masks[i]) == maxLength)
        
        
        print ("Excluded samples: " + str(excluded))
        return samples, labels, attention_masks
            
                
                
    def save(self):
        with open("train.json", "w") as json_file:
            json.dump({
                "samples":self.samples,
                "labels": self.labels,
                "att_masks":self.attention_masks}, json_file, indent=4)
        
        
        
        
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
    
    SEP_ID = tokenizer.convert_tokens_to_ids("[SEP]")
    CLS_ID = tokenizer.convert_tokens_to_ids("[CLS]")
    PAD_ID = tokenizer.convert_tokens_to_ids("[PAD]")

    tokenizer.add_tokens([B_TARGET, E_TARGET], special_tokens=True)
    tokenizer.add_tokens(list(EmotionsTokens.values()), special_tokens=True)


    B_TARGET_ID = tokenizer.convert_tokens_to_ids(B_TARGET)
    E_TARGET_ID = tokenizer.convert_tokens_to_ids(E_TARGET)
    EmotionsTokensId = {'anger':tokenizer.convert_tokens_to_ids('[ANGER]'), 
                      'disgust':tokenizer.convert_tokens_to_ids('[DISGUST]'),
                      'fear':tokenizer.convert_tokens_to_ids('[FEAR]'), 
                      'joy':tokenizer.convert_tokens_to_ids('[JOY]'),
                      'sadness':tokenizer.convert_tokens_to_ids('[SAD]'),
                      'surprise':tokenizer.convert_tokens_to_ids('[SURPISE]'),
                      'neutral': tokenizer.convert_tokens_to_ids('[NEUTRAL]')}
    
    #resize model w/ new tokens 
    model.resize_token_embeddings(len(tokenizer))
    
    s=Dataset('semEval-2024_CSCI-LING-5832-001/data/Subtask_1_1_train.json',
              None)
    
    
    input_ids = torch.tensor(s.samples) 
    attention_masks = torch.tensor(s.attention_masks) 
    
    dataset = TensorDataset(input_ids, attention_masks)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()  
    
    embeddings = []
    with torch.no_grad():  
        for batch in dataloader:
            input_ids, masks = batch
            outputs = model(input_ids, attention_mask=masks)
            last_hidden_states = outputs.last_hidden_state
            embeddings.append(last_hidden_states)
    
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, 'embeddings_tensor.pt')
    