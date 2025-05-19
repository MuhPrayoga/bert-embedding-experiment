import torch
from transformers import BertTokenizer, BertModel
from datasets.load_datasets import load_imdb, load_snli
from embeddings.extract_embeddings import extract_embedding
from models.classifier import SimpleClassifier
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def tokenize_batch(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

def get_embedding_dimension(strategy):
    return 768 if strategy != "concat_last_four" else 768 * 4

def train_and_evaluate(task_name, dataset, strategy):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model.eval()

    if task_name == "imdb":
        label_key = "label"
        text_key = "text"
        num_labels = 2
    elif task_name == "snli":
        label_key = "label"
        text_key = ("premise", "hypothesis")
        num_labels = 3

    # Ambil subset kecil untuk eksperimen cepat
    dataset = dataset["train"].filter(lambda x: x[label_key] != -1).select(range(2000))
    X, y = [], []

    for batch in tqdm(dataset, desc=f"Processing {strategy}"):
        if task_name == "imdb":
            inputs = tokenize_batch(batch[text_key], tokenizer)
        else:
            sent = batch[text_key[0]] + " [SEP] " + batch[text_key[1]]
            inputs = tokenize_batch(sent, tokenizer)

        with torch.no_grad():
            outputs = model(**inputs)
            layer_outputs = outputs.hidden_states
            embedding = extract_embedding(outputs, strategy, layer_outputs)
            cls_embedding = embedding[:, 0, :].squeeze()
            X.append(cls_embedding)
            y.append(batch[label_key])

    X = torch.stack(X)
    y = torch.tensor(y)
    clf = SimpleClassifier(X.shape[1], num_labels)
    optim = torch.optim.Adam(clf.parameters(), lr=2e-5)

    for epoch in range(3):
        clf.train()
        optim.zero_grad()
        pred = clf(X)
        loss = cross_entropy(pred, y)
        loss.backward()
        optim.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(X).argmax(dim=1)
        acc = accuracy_score(y, preds)
    print(f"{task_name.upper()} - {strategy} - Accuracy: {acc:.4f}")

if __name__ == "__main__":
    imdb = load_imdb()
    snli = load_snli()

    strategies = [
        "first", "last_hidden", "sum_all", "second_to_last", "sum_last_four", "concat_last_four"
    ]

    for strategy in strategies:
        train_and_evaluate("imdb", imdb, strategy)
        train_and_evaluate("snli", snli, strategy)
