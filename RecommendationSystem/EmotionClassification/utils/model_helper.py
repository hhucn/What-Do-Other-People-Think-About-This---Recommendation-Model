import torch


def load_model(model, checkpoint_path):
    print("Load model")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
        device = torch.device("cuda")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        device = torch.device("cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
