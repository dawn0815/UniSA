import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from baseline import erc_calculate,shot_calculate,msa_calculate,erc_score,data_ans,dot_process,golden
class CustomDataset(Dataset):
    def __init__(self, data ,tokenizer, max_length=128):
        self.data = data
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['task_type']+"<data>"+item['data_id']+"</data>"+"<ans>"+data_ans[item['data_id']]+"</ans>"+item['text']
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

data_dir='/mnt/workspace/model8_pretrain/'
train_data=load_data(data_dir+'datasets/humor_few_train.pkl',data_id='shot')
test_data=load_data(data_dir+'datasets/humor_few_test.pkl',data_id='shot')

tokenizer = T5Tokenizer.from_pretrained("/mnt/ruirenlzj/ruiren/t5_base_finetuned_best")

train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(test_data, tokenizer)

train_batch_size = 32
val_batch_size = 48

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

model = T5ForConditionalGeneration.from_pretrained("/mnt/ruirenlzj/ruiren/t5_base_finetuned_best")
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


num_epochs = 10
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
        model.save_pretrained("/mnt/workspace/model8_pretrain/t5_base_fewshot_best")
        tokenizer.save_pretrained("/mnt/workspace/model8_pretrain/t5_base_fewshot_best")
model_path = "/mnt/workspace/model8_pretrain/t5_base_fewshot_best"
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
tokenizer = T5Tokenizer.from_pretrained(model_path)
test_dataset = CustomDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.eval()

shot_pre,shot_label=[],[]


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

shot_mf1,shot_wf1=shot_calculate(shot_pre,shot_label)


print('shot :',shot_mf1)





