import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from baseline import erc_calculate,daily_calculate,msa_calculate,erc_score,data_ans,dot_process,golden
class CustomDataset(Dataset):
    def __init__(self, data ,tokenizer, max_length=256):
        self.data = data
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    def context_split(self, c_num, c_index, c_list):
        if c_num == 1:
            return "nocontext"
        else:
            a = ""
            start = max(0, c_index - 5)  # start of context
            end = min(c_num, c_index + 6)  # end of context
            for i in range(start, end):
                if i==c_index:
                    a+='<sep>'
                else:
                    a += c_list[i]
        return a

    def __getitem__(self, idx):
        item = self.data[idx]
        if item['task_type']=='erc':
            c_num=len(item['context'])
            c_list=item['context']
            c_index=item['index']
            text = item['task_type']+"<data>"+item['data_id']+"</data>"+"<ans>"+data_ans[item['data_id']]+"</ans>"+"<text>"+item['text']+"</text>"+"<context>"+self.context_split(c_num,c_index,c_list)+"</context>"
        else:
            text = item['task_type']+"<data>"+item['data_id']+"</data>"+"<ans>"+data_ans[item['data_id']]+"</ans>"+"<text>"+item['text']+"</text>"
        label = str(item['label'])
        input_encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(label, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'data_id':item['data_id'],
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }
def load_data(path,data_id):
    with open(path,'rb') as f:
        dataset = pickle.load(f)
    f.close()
    for d in dataset:
        d['data_id']=data_id
    return dataset

# data_dir='/root/data/ruiren/'
data_dir='/mnt/workspace/model8_pretrain/'
iemocap_train=load_data(data_dir+'datasets/iemocap_data_1124_pretrain.pkl',data_id='iemocap')
meld_train=load_data(data_dir+'datasets/meld_data_1124_pretrain.pkl',data_id='meld')
daily_train=load_data(data_dir+'datasets/daily_data_1124_pretrain.pkl',data_id='daily')
emory_train=load_data(data_dir+'datasets/emory_data_1124_pretrain.pkl',data_id='emory')
emowoz_train=load_data(data_dir+'datasets/emowoz_data_1124_pretrain.pkl',data_id='emowoz')
mosi_train=load_data(data_dir+'datasets/mosi_data_1123_pretrain.pkl',data_id='mosi')
mosei_train=load_data(data_dir+'datasets/mosei_data_1123_pretrain.pkl',data_id='mosei')
sst_train=load_data(data_dir+'datasets/sst_pretrain.pkl',data_id='sst')
imdb_train=load_data(data_dir+'datasets/imdb_pretrain.pkl',data_id='imdb')
train_data=iemocap_train+meld_train+emory_train+emowoz_train+daily_train+mosi_train+mosei_train+sst_train+imdb_train

iemocap_val=load_data(data_dir+'datasets/iemocap_data_1124_val.pkl',data_id='iemocap')
meld_val=load_data(data_dir+'datasets/meld_data_1124_val.pkl',data_id='meld')
daily_val=load_data(data_dir+'datasets/daily_data_1124_val.pkl',data_id='daily')
emory_val=load_data(data_dir+'datasets/emory_data_1124_val.pkl',data_id='emory')
emowoz_val=load_data(data_dir+'datasets/emowoz_data_1124_val.pkl',data_id='emowoz')
mosi_val=load_data(data_dir+'datasets/mosi_data_1123_val.pkl',data_id='mosi')
mosei_val=load_data(data_dir+'datasets/mosei_data_1123_val.pkl',data_id='mosei')
sst_val=load_data(data_dir+'datasets/sst_val.pkl',data_id='sst')
imdb_val=load_data(data_dir+'datasets/imdb_val.pkl',data_id='imdb')
val_data=iemocap_val+meld_val+emory_val+emowoz_val+daily_val+mosi_val+mosei_val+sst_val+imdb_val
tokenizer = T5Tokenizer.from_pretrained("t5-base")

train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)

train_batch_size = 16
val_batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

model = T5ForConditionalGeneration.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    step = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        # 输出每个step的损失值
        print(f"Step {step} - Train Loss: {loss.item()}")

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    step = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]

            total_loss += loss.item()
            step += 1

            # 输出每个step的损失值
            print(f"Step {step} - Validation Loss: {loss.item()}")

    return total_loss / len(dataloader)


num_epochs = 7
best_val_loss = float("inf")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)

    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f"Train loss: {train_loss}")

    val_loss = validate_epoch(model, val_dataloader, device)
    print(f"Validation loss: {val_loss}")

    # 仅在验证损失更低时保存模型
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model.")
        best_val_loss = val_loss
        model.save_pretrained("/root/data/ruiren/t5_base_finetuned_best")
        tokenizer.save_pretrained("/root/data/ruiren/t5_base_finetuned_best")

iemocap_test=load_data(data_dir+'datasets/iemocap_data_1124_test.pkl',data_id='iemocap')
meld_test=load_data(data_dir+'datasets/meld_data_1124_test.pkl',data_id='meld')
daily_test=load_data(data_dir+'datasets/daily_data_1124_test.pkl',data_id='daily')
emory_test=load_data(data_dir+'datasets/emory_data_1124_test.pkl',data_id='emory')
emowoz_test=load_data(data_dir+'datasets/emowoz_data_1124_test.pkl',data_id='emowoz')
mosi_test=load_data(data_dir+'datasets/mosi_data_1123_test.pkl',data_id='mosi')
mosei_test=load_data(data_dir+'datasets/mosei_data_1123_test.pkl',data_id='mosei')
sst_test=load_data(data_dir+'datasets/sst_val.pkl',data_id='sst')
imdb_test=load_data(data_dir+'datasets/imdb_val.pkl',data_id='imdb')
test_data=iemocap_test+meld_test+emory_test+emowoz_test+daily_test+mosi_test+mosei_test+sst_test+imdb_test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/root/data/ruiren/t5_base_finetuned_best"
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
tokenizer = T5Tokenizer.from_pretrained(model_path)
test_dataset = CustomDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.eval()

emowoz_pre,emowoz_label,emowoz_loss=[],[],0
sst_pre,sst_label,sst_loss=[],[],0
imdb_pre,imdb_label,imdb_loss=[],[],0
meld_pre,meld_label,meld_loss=[],[],0
iemocap_pre,iemocap_label,iemocap_loss=[],[],0
daily_pre,daily_label,daily_loss=[],[],0
emory_pre,emory_label,emory_loss=[],[],0
mosi_pre,mosi_label,mosi_loss=[],[],0
mosei_pre,mosei_label,mosei_loss=[],[],0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        data_ids = batch['data_id']
        true_labels = [tokenizer.decode(true_label, skip_special_tokens=True) for true_label in labels]
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        for j in range(len(data_ids)):
            pre=tokenizer.decode(generated_ids[j], skip_special_tokens=True)
            if data_ids[j] in ["mosi", "mosei"]:
                pre = dot_process(pre)
                true_labels[j]=float(true_labels[j])
            else:
                pre = erc_score(pre, golden[data_ids[j]])
            eval(data_ids[j] + "_pre").append(pre)
            eval(data_ids[j] + "_label").append(true_labels[j])

emowoz_f1,emowoz_acc=erc_calculate(emowoz_pre,emowoz_label)
sst_f1,sst_acc=erc_calculate(sst_pre,sst_label)
imdb_f1,imdb_acc=erc_calculate(imdb_pre,imdb_label)
meld_f1,meld_acc=erc_calculate(meld_pre,meld_label)
emory_f1,emory_acc=erc_calculate(emory_pre,emory_label)
daily_mf1,daily_wf1=daily_calculate(daily_pre,daily_label)
iemocap_f1,iemocap_acc=erc_calculate(iemocap_pre,iemocap_label)
mosi_mae,mosi_corr,mosi_acc7,mosi_acc2,mosi_acc20,mosi_f1,mosi_f10=msa_calculate(mosi_pre,mosi_label)
mosei_mae,mosei_corr,mosei_acc7,mosei_acc2,mosei_acc20,mosei_f1,mosei_f10=msa_calculate(mosei_pre,mosei_label)      

print('iemocap :',iemocap_f1,'  ',iemocap_acc)
print('meld :',meld_f1,'  ',meld_acc)
print('emowoz :',emowoz_f1,'  ',emowoz_acc)
print('daily :',daily_mf1,'  ',daily_wf1)
print('emory :',emory_f1,'  ',emory_acc)
print('sst :',sst_f1,'  ',imdb_acc)
print('imdb :',imdb_f1,'  ',imdb_acc)
print('mosi :',mosi_mae,' ',mosi_corr,' ',mosi_acc7,' ',mosi_acc2,' ',mosi_acc20,' ',mosi_f1,' ',mosi_f10)
print('mosei :',mosei_mae,' ',mosei_corr,' ',mosei_acc7,' ',mosei_acc2,' ',mosei_acc20,' ',mosei_f1,' ',mosei_f10)



