import torch


checkpoint_path = 'D:/workspace/Unetpp_CAMVID_seg_weights/best/checkpoint_epoch_1946.pth'


checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())
state_dict_keys = list(checkpoint['state_dict'].keys())
print("State dict keys:")
for key in state_dict_keys:
    print(key)