import torch
from MLCN import EncoderModel
from Dataloader import load_data
from Train import train_model, test_model
import warnings

warnings.filterwarnings('ignore')

def classification(epochs, batch_size, learning_rate, weight_decay):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataloader, test_dataloader, label_adj = load_data(batch_size=batch_size)
    model = EncoderModel(54, label_adj)

    model = model.to(device)
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99))
    step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(epochs):
        train_loss, train_prec, train_ndcg, train_metrics = train_model(train_dataloader, device, model, optimizer, criterion)
        test_loss, test_prec, test_ndcg, test_metrics = test_model(test_dataloader, device, model, criterion)
        step_scheduler.step(train_loss)

        print("Epoch={:02d}".format(epoch + 1))
        print("train_loss={:.4f}".format(train_loss.item()),
              "p@1:%.4f" % (train_prec[0]), "p@3:%.4f" % (train_prec[2]), "p@5:%.4f" % (train_prec[4]),
              "n@3:%.4f" % (train_ndcg[2]), "n@5:%.4f" % (train_ndcg[4]))
        print("train_HL:%.4f" % (train_metrics[0]), "Mi-P:%.4f" % (train_metrics[1]), "Mi-R:%.4f" % (train_metrics[2]), "Mi-F1:%.4f" % (train_metrics[3]),
              "Ma-P:%.4f" % (train_metrics[4]), "Ma-R:%.4f" % (train_metrics[5]), "Ma-F1:%.4f" % (train_metrics[6]))
        print("**test_loss={:.4f}".format(test_loss.item()),
             "p@1:%.4f" % (test_prec[0]), "p@3:%.4f" % (test_prec[2]), "p@5:%.4f" % (test_prec[4]),
             "n@3:%.4f" % (test_ndcg[2]), "n@5:%.4f" % (test_ndcg[4]))
        print("**test_HL:%.4f" % (test_metrics[0]), "Mi-P:%.4f" % (test_metrics[1]), "Mi-R:%.4f" % (test_metrics[2]), "Mi-F1:%.4f" % (test_metrics[3]),
             "Ma-P:%.4f" % (test_metrics[4]), "Ma-R:%.4f" % (test_metrics[5]), "Ma-F1:%.4f" % (test_metrics[6]))

if __name__ == '__main__':
    epochs = 10
    learning_rate = 3e-5
    batch_size = 32
    weight_decay = 0.0
    classification(epochs, batch_size, learning_rate, weight_decay)
