import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NextCharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.dense(last)
        return logits


def preprocess_data(seq, seq_len):
    chars = sorted(list(set(seq)))

    char_to_int = {c:i for i, c in enumerate(chars)}
    int_to_char = {i:c for c, i in char_to_int.items()}
    vocab_size = len(chars)

    print(f"Vocab: {chars}")
    print(f"Vocab size: {vocab_size}")

    X_list, y_list = [], []

    for i in range(len(seq) - seq_len):
        seq_in = seq[i : i + seq_len]
        seq_out = seq[i + seq_len]
        X_list.append([char_to_int[c] for c in seq_in])
        y_list.append(char_to_int[seq_out])

    X = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1) # (N, seq_len, 1
    y = torch.tensor(y_list, dtype=torch.long)

    # normalize input
    X = X / float(vocab_size)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    return loader, char_to_int, int_to_char, vocab_size, chars


def generate(seed, model, char_to_int, int_to_char,
             seq_len, vocab_size, n_chars=60):
    model.eval()
    pattern = [char_to_int[c] for c in seed][-seq_len:]

    out_text = seed

    for _ in range(n_chars):
        x = torch.tensor(pattern, dtype=torch.float32).view(1, seq_len, 1) / float(vocab_size)
        with torch.inference_mode():
            logits = model(x)
            next_id = int(torch.argmax(logits, dim=1).item())

        out_text += int_to_char[next_id]
        pattern.append(next_id)
        pattern = pattern[1:]

    return out_text

def main():

    TEXT = "aaabbbcccdddeee"
    SEED = "abbbcccd"
    N_CHARS = 60
    SEQ_LEN = 8
    seq = TEXT.lower()

    loader, char_to_int, int_to_char, vocab_size, chars = preprocess_data(seq, SEQ_LEN)

    model = NextCharLSTM(input_size=1, hidden_size=256, output_size=vocab_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    EPOCHS = 200

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch} - Loss = {avg_loss:.4f}")

    out_text = generate(
        seed=SEED,
        model=model,
        char_to_int=char_to_int,
        int_to_char=int_to_char,
        seq_len=SEQ_LEN,
        vocab_size=vocab_size,
        n_chars=N_CHARS
    )

    print(f"Generated:\n    {out_text}")


if __name__ == "__main__":
    main()