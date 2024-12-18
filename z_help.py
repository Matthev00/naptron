import torch

moodel_path = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712_cocofmt/latest.pth'
# Wczytaj checkpoint
checkpoint = torch.load(moodel_path)

# Dodaj klucz 'meta' (przykładowe wartości)
checkpoint['meta'] = {
    'epoch': 0,  # Dodaj bieżącą epokę
    'iter': 0,   # Dodaj bieżącą iterację
    'hook_msgs': {}  # Dodaj puste wiadomości hook
}

# Zapisz poprawiony checkpoint
torch.save(checkpoint, moodel_path)
