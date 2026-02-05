import torch
from torch import nn
from torch.optim import Adam

train_data = torch.load("embeddings/train.pt")
val_data = torch.load("embeddings/val.pt")

X_train = train_data["embeddings"]
y_train = train_data["labels"]

X_val = val_data["embeddings"]
y_val = val_data["labels"]

classifier = nn.Linear(768, 2)
optimizer = Adam(classifier.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 10

for epoch in range(EPOCHS):
    classifier.train()
    logits = classifier(X_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    classifier.eval()
    with torch.no_grad():
        preds = torch.argmax(classifier(X_val), dim=1)
        acc = (preds == y_val).float().mean().item()

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {loss.item():.4f} | "
        f"Val Acc: {acc:.2f}"
    )

torch.save(classifier.state_dict(), "saved_models/classifier.pt")
print("Classifier trained FAST")
